import ast
import csv
from itertools import product
import itertools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tnqec'))

import galois
import numpy as np

from planqtn.poly import UnivariatePoly

from collections import defaultdict
import time
from typing import List, Set, Tuple
from galois import GF2
import pandas as pd
from planqtn.progress_reporter import TqdmProgressReporter
from planqtn.legos import Legos
from planqtn.networks.compass_code import CompassCodeConcatenateAndSparsifyTN, CompassCodeDualSurfaceCodeLayoutTN, CompassCodeRotatedTN

from planqtn.progress_reporter import DummyProgressReporter, ProgressReporter
from planqtn.stabilizer_tensor_enumerator import (
    _index_leg,
    StabilizerCodeTensorEnumerator,
    _index_legs,
)
from planqtn.parity_check import tensor_product

from compassCodes.compass_code import CompassCode
from planqtn.tensor_network import _PartiallyTracedEnumerator
from planqtn.linalg import rank

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val  # or return None, or raise, depending on your needs
    return val

def update_csv_row_pandas(node1, node2, updates, num_run, name, filename):
    df = pd.read_csv(filename, delimiter=';')

    df["node_idx1"] = df["node_idx1"].apply(safe_literal_eval)
    df["node_idx2"] = df["node_idx2"].apply(safe_literal_eval)

    mask = (df['node_idx1'] == node1) & (df['node_idx2'] == node2) & (df['num_run'] == num_run) & (df['representation'] == name)

    if mask.any():
        for key, value in updates.items():
            if(key == "cost"):
                if(value == df.loc[mask, "new_subrank_cost"].iloc[0]):
                    df = df[~mask] # delete rows where our cost matches since we don't care
                    continue            
            df.loc[mask, key] = value
    else:
        raise ValueError("Row not found for update.")

    df.to_csv(filename, sep=';', index=False)


def matching_stabilizers_merge(g1, g2):
    tensor_prod = tensor_product(g1, g2)

    A = GF2([[1, 1, 0, 0],
             [0, 0, 1, 1]])  # Shape: (2 x 4)

    projection = tensor_prod @ A.T
    return rank(projection)


def fast_matching_stabilizer_ratio(generators):
    """
    Compute the ratio of stabilizers that act identically on the qubits
    (i.e., XX, YY, ZZ, or II), using linear algebra over GF(2).
    """

    parity_constraints = np.array([
        [1, 1, 0, 0],  # x0 + x1 = 0
        [0, 0, 1, 1]   # z0 + z1 = 0
    ], dtype=int)

    
    parity_vector = GF2(generators) @ GF2(parity_constraints.T % 2) 
    return rank(parity_vector)


def find_matching_stabilizers_ratio(generators):
    """
    Given k x 2n binary matrix (symplectic form),
    return all 2^k binary symplectic Pauli vectors (as numpy array).
    
    Each row in the output is a 2n vector: [x0, ..., xn-1, z0, ..., zn-1]
    """
    matches = 0
    basis = np.array(GF2(generators).row_space())
    r, n2 = basis.shape
    n = n2 // 2

    stabilizers = np.zeros((2**r, n2), dtype=int)
    for i, bits in enumerate(product([0, 1], repeat=r)):
        combo = np.zeros(n2, dtype=int)
        for j, b in enumerate(bits):
            if b:
                combo ^= basis[j]  # GF(2) addition
        stabilizers[i] = combo

        x = combo[:n]
        z = combo[n:]
        x0, x1 = x[0], x[1]
        z0, z1 = z[0], z[1]
        if (x0 == x1) and (z0 == z1):
            matches += 1

    return matches / len(stabilizers)

def stabilizer_enumerator_polynomial(
        tn,
        filename,
        num_run,
        name,
        open_legs: List[Tuple[int, int]] = [],
        verbose: bool = False,
        progress_reporter: ProgressReporter = DummyProgressReporter(),
        cotengra: bool = True,
        
    ):
        if tn._wep is not None:
            return tn._wep, contraction_width, contraction_cost

        assert (
            progress_reporter is not None
        ), "Progress reporter must be provided, it is None"

        with progress_reporter.enter_phase("collecting legs"):
            free_legs, leg_indices, index_to_legs = tn._collect_legs()

        open_legs_per_node = defaultdict(list)
        for node_idx, node in tn.nodes.items():
            for leg in node.legs:
                if leg not in free_legs:
                    open_legs_per_node[node_idx].append(_index_leg(node_idx, leg))

        for node_idx, leg_index in open_legs:
            open_legs_per_node[node_idx].append(_index_leg(node_idx, leg_index))

        intermediate_tensor_sizes = []

        for node_idx, node in tn.nodes.items():
            traced_legs = open_legs_per_node[node_idx]      
        
            tensor = node.stabilizer_enumerator_polynomial(
                open_legs=traced_legs,
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )
            if len(traced_legs) == 0:
                tensor = {(): tensor}
            tn._ptes[node_idx] = _PartiallyTracedEnumerator(
                nodes={node_idx},
                tracable_legs=open_legs_per_node[node_idx],
                tensor=tensor,  
                truncate_length=tn.truncate_length,
            )

            intermediate_tensor_sizes.append(len(tensor))
        if cotengra and len(tn.nodes) > 0 and len(tn._traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, open_legs_per_node, verbose, progress_reporter
                )
            contraction_width = tree.contraction_width()
            contraction_cost = tree.contraction_cost()
            score_cotengra = tree.get_score()
            total_flops = tree.total_flops()
            
            subrank_w_self_trace = test_conjoining_cost(tn, traces, num_run, name, filename, open_legs_per_node)

        if len(traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        total_ops_count = 0
        num_mismatches_merge = 0
        num_mismatches_self = 0

        open_legs_dict = {}
        for node_idx, pte in tn._ptes.items():
            open_legs_dict[node_idx] = len(pte.tracable_legs) - 1

        nodes_pcm = list(tn.nodes.values())
        ptes_pcm: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
            (node, {node.tensor_id}) for node in nodes_pcm
        ]
        node_to_pte = {node.tensor_id: i for i, node in enumerate(nodes_pcm)}
        
        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn._ptes else tn._ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn._ptes else tn._ptes[node_idx2]

            join_legs1 = _index_legs(node_idx1, join_legs1)
            join_legs2 = _index_legs(node_idx2, join_legs2)

            pte1_pcm_idx = node_to_pte.get(node_idx1)
            pte2_pcm_idx = node_to_pte.get(node_idx2)

            open_legs1_est = open_legs_dict.get(node_idx1, 0)
            open_legs2_est = open_legs_dict.get(node_idx2, 0)

            if node1_pte == node2_pte:
                print("Self tracing pte: ", node_idx1)
                #print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
                # both nodes are in the same PTE!
                pte, ops_count = node1_pte.self_trace(
                    join_legs1=[
                        (node_idx1, leg) if isinstance(leg, int) else leg
                        for leg in join_legs1
                    ],
                    join_legs2=[
                        (node_idx2, leg) if isinstance(leg, int) else leg
                        for leg in join_legs2
                    ],
                    progress_reporter=progress_reporter,
                    verbose=verbose,
                )

                for node in pte.nodes:
                    tn._ptes[node] = pte
                tn._legs_left_to_join[node_idx1] = [
                    leg
                    for leg in tn._legs_left_to_join[node_idx1]
                    if leg not in join_legs1
                ]
                tn._legs_left_to_join[node_idx2] = [
                    leg
                    for leg in tn._legs_left_to_join[node_idx2]
                    if leg not in join_legs2
                ]
                intermediate_tensor_sizes.append(len(pte.tensor))

                prev_pte_pcm, nodes_pcm = ptes_pcm[pte1_pcm_idx]
                new_pte_pcm = prev_pte_pcm.self_trace(join_legs1, join_legs2)
                ptes_pcm[pte1_pcm_idx] = (new_pte_pcm, nodes_pcm)

                for node in nodes_pcm:
                    open_legs_dict[node] = open_legs1_est - 2

                open_legs1_est -= 1
                open_legs2_est -= 1

                open_legs_set = set(pte.tracable_legs)

                open_leg_indices = [i for i, leg in enumerate(new_pte_pcm.legs) if leg in open_legs_set]
                open_leg_indices += [i + (new_pte_pcm.h.shape[1]//2) for i, leg in enumerate(new_pte_pcm.legs) if leg in open_legs_set]
                open_leg_submatrix = new_pte_pcm.h[:, open_leg_indices]
                rank_submatrix = rank(open_leg_submatrix)


                prev_legs_set = set(node1_pte.tracable_legs)
                prev_open_leg_indices = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in prev_legs_set]
                prev_open_leg_indices += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in prev_legs_set]
                prev_open_leg_submatrix = prev_pte_pcm.h[:, prev_open_leg_indices]
                prev_rank_submatrix = rank(prev_open_leg_submatrix)


                join_idxs = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs1]
                join_idxs += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs1]
                join_idxs2 = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs2]
                join_idxs2 += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs2]
                joined1 = prev_pte_pcm.h[:, [join_idxs[0], join_idxs2[0], join_idxs[1], join_idxs2[1]]]
                
                join_rank = fast_matching_stabilizer_ratio(joined1)
                r_join_stack = rank(joined1)

                matches = find_matching_stabilizers_ratio(joined1)
                print("matches ratio is: ", matches)

                if(2**(prev_rank_submatrix)*matches != ops_count):
                    num_mismatches_self += 1
                    print("\t RANK != COST")
                    #     print("Open legs mismatch, my est: ", open_legs1, open_legs2, " vs ", open_legs)
                
                    print("\t  self traced pte, new length: ", len(pte.tensor))
                    #print("\t new parity check matrix: ", new_pcm_pte.h)
                    print("\t  original length: ", len(node1_pte.tensor))
                    #print("\t self traced pte parity check matrix: ", new_pte_pcm.h)
                    print("\t  cost: ", ops_count)
                    print("\t submatrix rank: ", rank_submatrix)
                    print("\t number of open legs: ", len(pte.tracable_legs))
                    #print("\t number of cols: ", new_pcm_pte.h.shape[1]//2)
                    #print("\t legs of matrix: ", new_pte_pcm.legs)
                    print("\t previous submatrix rank: ", prev_rank_submatrix)
                    print("\t rank of full previous h: ", rank(prev_pte_pcm.h))
                    #print("\t previous pcm legs: ", prev_pte_pcm.legs)
                    #print("\t previous pcm: ", prev_pte_pcm.h)
                    print("\t join legs 1: ", join_legs1)
                    print("\t join legs 2: ", join_legs2)
                    #print("\t previous open legs: ", prev_legs_set)

                    print("\t joined legs matrix: ", joined1)
                    parity_constraints = np.array([
                        [1, 1, 0, 0],  # x0 + x1 = 0
                        [0, 0, 1, 1]   # z0 + z1 = 0
                    ], dtype=int)

                    
                    parity_vector = GF2(joined1) @ GF2(parity_constraints.T % 2)
                    print("parity vector is: ", parity_vector)
                    print("rank of parity vector is: ", rank(parity_vector)) 

                # print("\t  open legs1: ", open_legs)
                # print("\t  pte sparsity: ", len(node1_pte.tensor) / (4**open_legs))
                #update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : len(open_legs), "open_legs2": len(open_legs), "rank_submatrix1" : prev_rank_submatrix, "rank_submatrix2" : -1, "r_join_stack/keys_skipped" : join_rank}, num_run, name, filename)

            else:
                pte, ops_count = node1_pte.merge_with(
                    node2_pte,
                    join_legs1=[
                        (node_idx1, leg) if isinstance(leg, int) else leg
                        for leg in join_legs1
                    ],
                    join_legs2=[
                        (node_idx2, leg) if isinstance(leg, int) else leg
                        for leg in join_legs2
                    ],
                    verbose=verbose,
                    progress_reporter=progress_reporter,
                )

                pte1_pcm, nodes1_pcm = ptes_pcm[pte1_pcm_idx]
                pte2_pcm, nodes2_pcm = ptes_pcm[pte2_pcm_idx]
                new_pcm_pte = pte1_pcm.conjoin(pte2_pcm, legs1=join_legs1, legs2=join_legs2)
                merged_nodes_pcm = nodes1_pcm.union(nodes2_pcm)

                open_legs_dict[node_idx1] = open_legs1_est + open_legs2_est - 1
                for node in merged_nodes_pcm:
                    open_legs_dict[node] = open_legs1_est + open_legs2_est - 1

                open_legs_set = set(pte.tracable_legs)
                open_leg_indices = [i for i, leg in enumerate(new_pcm_pte.legs) if leg in open_legs_set]
                open_leg_indices += [i + (new_pcm_pte.h.shape[1]//2) for i, leg in enumerate(new_pcm_pte.legs) if leg in open_legs_set]
                open_leg_submatrix = new_pcm_pte.h[:, open_leg_indices]
                rank_submatrix = rank(open_leg_submatrix)

                open_legs_set1 = set(node1_pte.tracable_legs)
                open_leg_indices1 = [i for i, leg in enumerate(pte1_pcm.legs) if leg in open_legs_set1]
                open_leg_indices1 += [i + (pte1_pcm.h.shape[1]//2) for i, leg in enumerate(pte1_pcm.legs) if leg in open_legs_set1]
                open_leg_submatrix1 = pte1_pcm.h[:, open_leg_indices1]
                rank_submatrix1 = rank(open_leg_submatrix1)

                open_legs_set2 = set(node2_pte.tracable_legs)
                open_leg_indices2 = [i for i, leg in enumerate(pte2_pcm.legs) if leg in open_legs_set2]
                open_leg_indices2 += [i + (pte2_pcm.h.shape[1]//2) for i, leg in enumerate(pte2_pcm.legs) if leg in open_legs_set2]
                open_leg_submatrix2 = pte2_pcm.h[:, open_leg_indices2]
                rank_submatrix2 = rank(open_leg_submatrix2)

                join_idxs2 = [i for i, leg in enumerate(pte2_pcm.legs) if leg in join_legs2]
                join_idxs2 += [i + (pte2_pcm.h.shape[1]//2) for i, leg in enumerate(pte2_pcm.legs) if leg in join_legs2]
                join_idxs = [i for i, leg in enumerate(pte1_pcm.legs) if leg in join_legs1]
                join_idxs += [i + (pte1_pcm.h.shape[1]//2) for i, leg in enumerate(pte1_pcm.legs) if leg in join_legs1]
                joined1 = pte1_pcm.h[:, join_idxs]
                joined2 = pte2_pcm.h[:, join_idxs2]

                stacked = np.vstack([joined1 & 1, joined2 & 1])  
                r_join_stack = rank(stacked)
                     
                # Update the first PTE with merged result
                ptes_pcm[pte1_pcm_idx] = (new_pcm_pte, merged_nodes_pcm)
                # Remove the second PTE
                ptes_pcm.pop(pte2_pcm_idx)

                # Update node_to_pte mappings
                for node in nodes2_pcm:
                    node_to_pte[node] = pte1_pcm_idx
                # Adjust indices for all nodes in PTEs after the removed one
                for node, pte_idx in node_to_pte.items():
                    if pte_idx > pte2_pcm_idx:
                        node_to_pte[node] = pte_idx - 1

                
                for node in pte.nodes:
                    tn._ptes[node] = pte
                tn._legs_left_to_join[node_idx1] = [
                    leg
                    for leg in tn._legs_left_to_join[node_idx1]
                    if leg not in join_legs1
                ]
                tn._legs_left_to_join[node_idx2] = [
                    leg
                    for leg in tn._legs_left_to_join[node_idx2]
                    if leg not in join_legs2
                ]
                intermediate_tensor_sizes.append(len(pte.tensor))

                #join_rank = matching_stabilizers_merge(joined1, joined2)
                
 
                if(r_join_stack != 2):
                    print("NOT EQUAL TO 2 during merge!, r_join_stack is: ", r_join_stack)

                tensor_prod = tensor_product(joined1, joined2)                
                matches = find_matching_stabilizers_ratio(tensor_prod)

                # Check if any row matches the target
                if(2**(rank_submatrix1 + rank_submatrix2)*matches != ops_count):
                    num_mismatches_merge += 1
                    print("\t RANK != COSt")
                    print("MERGING PTEs: ", node_idx1, node_idx2)
                    print("\tnew pte length: ", len(pte.tensor), " full elements would be: ", 4**(open_legs1_est + open_legs2_est))

                    print("tensor product: ", tensor_prod)
                    print("\t joined 1 matrix: ", joined1)
                    print("\t joined 2 matrix: ", joined2)
                    
                    
                    # print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
                    # print("\t open legs1: ", open_legs1, " open legs 2: ", open_legs2)
                    # print("\t  merged pte, new length: ", len(pte.tensor))
                    print("\t  original lengths: ", len(node1_pte.tensor), len(node2_pte.tensor))
                    #print("\t estimated original lengths: ", estimated_lengths[node_idx1], estimated_lengths[node_idx2])
                    # #print("\t  merged pte parity check matrix: ", pte.h)
                    print("\t  cost: ", ops_count)
                        
                        
                    print("\t submatrix rank: ", rank_submatrix)
                    print("\t rank of node 1: ", rank_submatrix1)
                    print("\t rank of node 2: ", rank_submatrix2)
                    print("\t rank of joined stacked: ", r_join_stack)
                    #print("\t joined 1 matrix: ", joined1)
                    #print("\t joined 2 matrix: ", joined2)
               
                
                    
                #update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : open_legs1_est, "open_legs2" : open_legs2_est, "rank_submatrix1" : rank_submatrix1, "rank_submatrix2" : rank_submatrix2, "r_join_stack/keys_skipped" : r_join_stack}, num_run, name, filename)

            total_ops_count += ops_count
            node1_pte = None if node_idx1 not in tn._ptes else tn._ptes[node_idx1]

            for k in list(node1_pte.tensor.keys()):
                v = node1_pte.tensor[k]
                if tn.truncate_length is None:
                    continue
                if v.minw()[0] > tn.truncate_length:
                    del pte.tensor[k]
                else:
                    pte.tensor[k].truncate_inplace(tn.truncate_length)

        if len(set(tn._ptes.values())) > 1:
            pte_list = list(set(tn._ptes.values()))
            pte = pte_list[0]
            for pte2 in pte_list[1:]:
                pte, ops1, ops2 = pte.tensor_product(pte2, verbose=verbose)

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, total_ops_count, contraction_cost, total_flops, score_cotengra, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge


def get_contraction_time(tn, cotengra, num_run, name, filename):
    tn.analyze_traces(cotengra=cotengra, on_trial_error='raise', max_repeats=300)
    start = time.time()
    wep, total_ops_count, contraction_cost, total_flops, score_cotengra, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge = stabilizer_enumerator_polynomial(
        tn, filename, num_run, name, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=cotengra
    )
    end = time.time()
    return end - start, wep, total_ops_count, contraction_cost, total_flops, score_cotengra, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge

def get_rank_for_matrix_legs(pte, open_legs):
    open_legs_set = set(open_legs)
    open_leg_indices = [i for i, leg in enumerate(pte.legs) if leg in open_legs_set]
    open_leg_indices += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in open_legs_set]
    open_leg_submatrix = pte.h[:, open_leg_indices]
    return  rank(open_leg_submatrix)

def test_conjoining_cost(tn, traces, num_run, name, filename, open_legs_per_node):
    # If there's only one node and no traces, return it directly
    if len(tn.nodes) == 1 and len(tn._traces) == 0:
        return list(tn.nodes.values())[0]
    
    join_legs_cost = 0

    open_legs = {}
    traceable_legs = {}
    for node_idx, legs in open_legs_per_node.items():
        open_legs[node_idx] = len(legs) - 1
        traceable_legs[node_idx] = legs

    # Map from node_idx to the index of its PTE in ptes list
    nodes = list(tn.nodes.values())
    ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
        (node, {node.tensor_id}) for node in nodes
    ]
    node_to_pte = {node.tensor_id: i for i, node in enumerate(nodes)}

    for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
        join_legs1 = _index_legs(node_idx1, join_legs1)
        join_legs2 = _index_legs(node_idx2, join_legs2)

        pte1_idx = node_to_pte.get(node_idx1)
        pte2_idx = node_to_pte.get(node_idx2)

        open_legs1 = open_legs.get(node_idx1, 0)
        open_legs2 = open_legs.get(node_idx2, 0)


        # Case 1: Both nodes are in the same PTE
        if pte1_idx == pte2_idx:
            pte, nodes = ptes[pte1_idx]

            new_traceable_legs = [
                leg
                for leg in traceable_legs[node_idx1]
                if leg not in join_legs1 and leg not in join_legs2
            ]

            for node in nodes:
                open_legs[node] = open_legs1 - 2
                traceable_legs[node] = new_traceable_legs

            open_legs1 -= 1
            open_legs2 -= 1
            
            new_pte = pte.self_trace(join_legs1, join_legs2)
            ptes[pte1_idx] = (new_pte, nodes)

            rank_submatrix = get_rank_for_matrix_legs(new_pte, new_traceable_legs)
            prev_rank_submatrix = get_rank_for_matrix_legs(pte, new_traceable_legs + join_legs1 + join_legs2)

            join_idxs = [i for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs2 = [i for i, leg in enumerate(pte.legs) if leg in join_legs1]
            join_idxs2 += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs1]
            joined1 = pte.h[:, [join_idxs[0], join_idxs2[0], join_idxs[1], join_idxs2[1]]]

            matches = find_matching_stabilizers_ratio(joined1)
            join_legs_cost += 2**(prev_rank_submatrix) * matches

            print("self tracing nodes: ", node_idx1)
            print("join legs are: ", join_legs1, join_legs2)
            print("cost of operation is: ", 2**(prev_rank_submatrix) * matches)
            # matches = fast_matching_stabilizer_ratio(joined1)
            # join_legs_cost += 2**(prev_rank_submatrix - matches)


            # with open(filename, 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter=';')
            #     writer.writerow([num_run, name, node_idx1, node_idx2, True, pte.h.shape, -1, new_pte.h.shape, open_legs1, open_legs2, rank_submatrix, prev_rank_submatrix, -1, join_rank, 2**(prev_rank_submatrix)*join_rank])

        # Case 2: Nodes are in different PTEs - merge them
        else:
            open_legs[node_idx1] = open_legs1 + open_legs2 - 1

            pte1, nodes1 = ptes[pte1_idx]
            pte2, nodes2 = ptes[pte2_idx]

            new_pte = pte1.conjoin(pte2, legs1=join_legs1, legs2=join_legs2)
            merged_nodes = nodes1.union(nodes2)
            # Update the first PTE with merged result
            ptes[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            ptes.pop(pte2_idx)

            new_traceable_legs = [
                leg
                for leg in traceable_legs[node_idx1]
                if leg not in join_legs1 and leg not in join_legs2
            ]

            new_traceable_legs += [
                leg
                for leg in traceable_legs[node_idx2]
                if leg not in join_legs1 and leg not in join_legs2
            ]
            rank_submatrix1 = get_rank_for_matrix_legs(pte1, traceable_legs[node_idx1])
            rank_submatrix2 = get_rank_for_matrix_legs(pte2, traceable_legs[node_idx2])

            join_idxs2 = [i for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs2 += [i + (pte2.h.shape[1]//2) for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs = [i for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            join_idxs += [i + (pte1.h.shape[1]//2) for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            joined1 = pte1.h[:, join_idxs]
            joined2 = pte2.h[:, join_idxs2]

            tensor_prod = tensor_product(joined1, joined2)                
            matches = find_matching_stabilizers_ratio(tensor_prod)
            join_legs_cost += (2**(rank_submatrix1 + rank_submatrix2)* matches)
            # matches = matching_stabilizers_merge(joined1, joined2)
            # join_legs_cost += (2**(rank_submatrix1 + rank_submatrix2 - matches))

            print("merging nodes: ", node_idx1, node_idx2)
            print("join legs are: ", join_legs1, join_legs2)
            print("cost of operation is: ", (2**(rank_submatrix1 + rank_submatrix2)* matches))

            for node in merged_nodes:
                open_legs[node] = open_legs1 + open_legs2 - 1
                traceable_legs[node] = new_traceable_legs

            # Update node_to_pte mappings
            for node in nodes2:
                node_to_pte[node] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node] = pte_idx - 1
     
            # with open(filename, 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter=';')
            #     writer.writerow([num_run, name, node_idx1, node_idx2, False, pte1.h.shape, pte2.h.shape, new_pte.h.shape, open_legs1, open_legs2, rank_submatrix, rank_submatrix1, rank_submatrix2, r_join_stack, (2**(rank_submatrix1 + rank_submatrix2 - r_join_stack))])
        
    return join_legs_cost


def run_wep_calc(q_shor, num_runs, coloring, distance, file_name):
    # for name, rep in compass_code.get_representations().items():

    compass_code_shor = CompassCode(distance, [[2,2], [2,2]])
    tn_concat = compass_code_shor.concatenated()
    for i in range(num_runs):
        contraction_time, wep, total_ops_count, contraction_cost, total_flops, score_cotengra, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge = get_contraction_time(tn_concat, True, i, "concat", file_name)

        with open("tn_architectures_cost.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([i, "Concatenated", distance, contraction_time, total_ops_count, score_cotengra, subrank_w_self_trace, contraction_cost, total_flops])




filename = 'truncated_steps.csv'

# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     writer.writerow(["num_run", "representation", "node_idx1", "node_idx2", "self trace?",
#                      "pcm_1 size", "pcm_2 size", "new pcm size",  
#                      "open_legs1 est", "open_legs2 est", "rank_submatrix",
#                      "rank_submatrix1_est", "rank_submatrix2_est", "r_join_stack_est", "new_subrank_cost",
#                      "pte 1 len", "pte 2 len", "new pte len", "cost", "open_legs1", "open_legs2", "rank_submatrix1", "rank_submatrix2", "r_join_stack/keys_skipped"])


# with open("optimal_tests.csv", 'w', newline='') as csvfile:
#      writer = csv.writer(csvfile, delimiter=';')
#      writer.writerow(["num_run", "representation", "distance", "contraction_time", "total_ops_count", "score_cotengra", "custom_subrank_cost", "contraction_cost", "total_flops_cotengra"])


def generate_checkerboard_coloring(d):
    return np.array([[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)])


# ds = [3]
# q_shor = 0.0

# for d in ds:
#     coloring = generate_checkerboard_coloring(d)
#     run_wep_calc(q_shor, 1, coloring, d, filename)


d = 3
coloring = np.full((d, d), 2)
compass_code = CompassCode(d, coloring)
tn_concat = compass_code.concatenated()
contraction_time, wep, total_ops_count, contraction_cost, total_flops, score_cotengra, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge = get_contraction_time(tn_concat, True, 0, "concat", "t")



def run_test(coloring):
    coset = GF2([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    coset = None
    tn_concat = CompassCodeConcatenateAndSparsifyTN(
        coloring,
        coset_error=coset,
        truncate_length=2,
    )

    wep_concat = tn_concat.stabilizer_enumerator_polynomial(cotengra=True)
    print("concatenate and sparsity wep: ", wep_concat)

    for q in range(tn_concat.n_qubits()):
        node, (nid, leg) = tn_concat.qubit_to_node_and_leg(q)
        print(f"Qubit {q} -> Node {node}, Leg {leg}")
    
    tn_dual = CompassCodeDualSurfaceCodeLayoutTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512,
        coset_error=coset,
        truncate_length=2,
    )

    wep_dual = tn_dual.stabilizer_enumerator_polynomial(cotengra=True)
    # print("tn_dual parity check:")
    # print(tn_dual.conjoin_nodes().h)
    print("dual surface wep: ", wep_dual)

    for q in range(tn_dual.n_qubits()):
        node, (nid, leg) = tn_dual.qubit_to_node_and_leg(q)
        print(f"Qubit {q} -> Node {node}, Leg {leg}")

    tn_rotated = CompassCodeRotatedTN(
        coloring,
        lego=lambda i: Legos.encoding_tensor_512,
        coset_error=coset,
        truncate_length=2,
    )
    wep_rotated = tn_rotated.stabilizer_enumerator_polynomial(cotengra=True)
    print("rotated surface wep: ", wep_dual)
    assert wep_concat == wep_dual == wep_rotated

# assert wep == UnivariatePoly({4: 10, 6: 5, 2: 1}), f"Not equal, got:\n{wep}"

# coloring_len = 3
# for combo in itertools.product([1, 2], repeat=coloring_len**2):
#     matrix = np.array(combo).reshape((coloring_len, coloring_len))
#     print("running for coloring: ")
#     print(matrix)
#     run_test(matrix) 

# coloring = np.array([[1, 1], [2, 1]])
# result = run_test(coloring) 