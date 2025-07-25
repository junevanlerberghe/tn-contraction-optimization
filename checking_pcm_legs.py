import ast
import csv
import math
import random

import galois
import numpy as np
from collections import defaultdict
import time
from typing import List, Set, Tuple
from galois import GF2
import pandas as pd
import scipy
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.stabilizer_tensor_enumerator import (
    _index_leg,
    StabilizerCodeTensorEnumerator,
    _index_legs,
)

from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator
from qlego.symplectic import sslice

def gf2_rank(matrix: np.ndarray) -> int:
    """
    Computes the rank of a binary matrix over GF(2) using Gaussian elimination.
    """
    matrix = np.array(matrix, dtype=int) % 2
    A = matrix.copy()
    n_rows, n_cols = A.shape
    rank = 0
    col = 0

    for row in range(n_rows):
        while col < n_cols and A[row:, col].max() == 0:
            col += 1
        if col == n_cols:
            break
        pivot_row = row + np.argmax(A[row:, col])
        A[[row, pivot_row]] = A[[pivot_row, row]]
        for r in range(n_rows):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        rank += 1
        col += 1

    return rank


def compute_gf2_rank(pcm_sub):
    GF2 = galois.GF(2)
    pcm_gf2 = GF2(pcm_sub)
    return pcm_gf2.row_space().shape[0]

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val  # or return None, or raise, depending on your needs
    return val

def update_csv_row_pandas(node1, node2, updates, num_run, name, filename):
    # Step 1: Load the CSV into a DataFrame
    df = pd.read_csv(filename, delimiter=';')

    df["node_idx1"] = df["node_idx1"].apply(safe_literal_eval)
    df["node_idx2"] = df["node_idx2"].apply(safe_literal_eval)

    # Step 2: Find the matching row using boolean indexing
    mask = (df['node_idx1'] == node1) & (df['node_idx2'] == node2) & (df['num_run'] == num_run) & (df['representation'] == name)

    # Step 3: Update the relevant columns
    if mask.any():
        for key, value in updates.items():
            df.loc[mask, key] = value
    else:
        raise ValueError("Row not found for update.")

    # Step 4: Write back to the CSV
    df.to_csv(filename, sep=';', index=False)


def stabilizer_enumerator_polynomial(
        tn,
        filename,
        num_run,
        name,
        open_legs: List[Tuple[int, int]] = [],
        verbose: bool = False,
        progress_reporter: ProgressReporter = DummyProgressReporter(),
        cotengra: bool = True,
        
    ) -> Tuple[SimplePoly, float, int, List[int]] :
        contraction_width = -1
        contraction_cost = -1

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

        traces = tn.traces

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
            tn.ptes[node_idx] = _PartiallyTracedEnumerator(
                nodes={node_idx},
                tracable_legs=open_legs_per_node[node_idx],
                tensor=tensor,  
                truncate_length=tn.truncate_length,
            )

            intermediate_tensor_sizes.append(len(tensor))
        if cotengra and len(tn.nodes) > 0 and len(tn.traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, verbose, progress_reporter
                )
            contraction_width = tree.contraction_width()
            contraction_cost = tree.contraction_cost()
            score_cotengra = tree.get_score()
            
            final_pcm, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, total_submatrix_rank_cost, subrank_w_self_trace = test_conjoining_cost(tn, traces, num_run, name, filename)

        summed_legs = [leg for leg in free_legs if leg not in open_legs]

        if len(tn.traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        total_ops_count = 0
        total_ops_count2 = 0
        num_mismatches_merge = 0
        num_mismatches_self = 0

        open_legs_dict = {}
        estimated_lengths = {}
        for node_idx, pte in tn.ptes.items():
            open_legs_dict[node_idx] = len(pte.tracable_legs) - 1
            estimated_lengths[node_idx] = len(pte.tensor)

        nodes_pcm = list(tn.nodes.values())
        ptes_pcm: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
            (node, {node.idx}) for node in nodes_pcm
        ]
        node_to_pte = {node.idx: i for i, node in enumerate(nodes_pcm)}


        print("GOING INTO ACTUAL CONTRACTION NOW")
        
        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn.ptes else tn.ptes[node_idx2]

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
                    tn.ptes[node] = pte
                tn.legs_left_to_join[node_idx1] = [
                    leg
                    for leg in tn.legs_left_to_join[node_idx1]
                    if leg not in join_legs1
                ]
                tn.legs_left_to_join[node_idx2] = [
                    leg
                    for leg in tn.legs_left_to_join[node_idx2]
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
                rank_submatrix = gf2_rank(open_leg_submatrix)


                prev_legs_set = set(node1_pte.tracable_legs)
                prev_open_leg_indices = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in prev_legs_set]
                prev_open_leg_indices += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in prev_legs_set]
                prev_open_leg_submatrix = prev_pte_pcm.h[:, prev_open_leg_indices]
                prev_rank_submatrix = gf2_rank(prev_open_leg_submatrix)


                join_idxs = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs1]
                join_idxs += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs1]
                join_idxs2 = [i for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs2]
                join_idxs2 += [i + (prev_pte_pcm.h.shape[1]//2) for i, leg in enumerate(prev_pte_pcm.legs) if leg in join_legs2]
                joined1 = prev_pte_pcm.h[:, join_idxs + join_idxs2]
                #print("\t join idxs1: ", join_idxs, " join idxs2: ", join_idxs2)
                joined_cols1 = prev_pte_pcm.h[:, join_idxs]
                joined2 = prev_pte_pcm.h[:, join_idxs2]
                
                stacked = np.vstack([joined_cols1 & 1, joined2 & 1])  
                r_join_stack = gf2_rank(joined1)
                r_stack_vert = gf2_rank(stacked)

                matching = np.all(joined_cols1 == joined2, axis=1)
                not_all_zeros = ~np.all(joined_cols1 == 0, axis=1)  # assumes you want to ignore if A is all 0s

                # Combine both conditions
                valid_matches = matching & not_all_zeros

                # Count
                matching_rows = np.sum(valid_matches)
                
                keys_skipped = (2**(rank_submatrix)) / (2**(prev_rank_submatrix))
                estimated_cost = (2**(prev_rank_submatrix)) * keys_skipped
                
                print("\tjoin rank: ", r_join_stack)
                # print("\tjoin rank vertical: ", r_stack_vert)
                # print("\tnum matching rows: ", matching_rows)
                # print("\tdiff btw prev rank and new:", prev_rank_submatrix - rank_submatrix)
                # print("\tprev rank:", prev_rank_submatrix)

                difference = r_join_stack - 2
                if(difference < 0):
                    difference = 0

                if(2**(prev_rank_submatrix - difference) != ops_count):
                    num_mismatches_self += 1
                    print("\t RANK != COST")
                if(r_join_stack == 2):
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
                    print("\t rank of full previous h: ", gf2_rank(prev_pte_pcm.h))
                    print("\t previous pcm legs: ", prev_pte_pcm.legs)
                    print("\t previous pcm: ", prev_pte_pcm.h)
                    print("\t join legs 1: ", join_legs1)
                    print("\t join legs 2: ", join_legs2)
                    print("\t previous open legs: ", prev_legs_set)

                for node in nodes_pcm:
                    estimated_lengths[node] = 2**rank_submatrix

                # print("\t  open legs1: ", open_legs)
                # print("\t  pte sparsity: ", len(node1_pte.tensor) / (4**open_legs))
                update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : len(open_legs), "open_legs2": len(open_legs), "pte1 sparsity" : len(node1_pte.tensor), "pte2 sparsity": len(node1_pte.tensor)}, num_run, name, filename)

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
                rank_submatrix = gf2_rank(open_leg_submatrix)

                open_legs_set1 = set(node1_pte.tracable_legs)
                open_leg_indices1 = [i for i, leg in enumerate(pte1_pcm.legs) if leg in open_legs_set1]
                open_leg_indices1 += [i + (pte1_pcm.h.shape[1]//2) for i, leg in enumerate(pte1_pcm.legs) if leg in open_legs_set1]
                open_leg_submatrix1 = pte1_pcm.h[:, open_leg_indices1]
                rank_submatrix1 = gf2_rank(open_leg_submatrix1)

                open_legs_set2 = set(node2_pte.tracable_legs)
                open_leg_indices2 = [i for i, leg in enumerate(pte2_pcm.legs) if leg in open_legs_set2]
                open_leg_indices2 += [i + (pte2_pcm.h.shape[1]//2) for i, leg in enumerate(pte2_pcm.legs) if leg in open_legs_set2]
                open_leg_submatrix2 = pte2_pcm.h[:, open_leg_indices2]
                rank_submatrix2 = gf2_rank(open_leg_submatrix2)

                join_idxs2 = [i for i, leg in enumerate(pte2_pcm.legs) if leg in join_legs2]
                join_idxs2 += [i + (pte2_pcm.h.shape[1]//2) for i, leg in enumerate(pte2_pcm.legs) if leg in join_legs2]
                join_idxs = [i for i, leg in enumerate(pte1_pcm.legs) if leg in join_legs1]
                join_idxs += [i + (pte1_pcm.h.shape[1]//2) for i, leg in enumerate(pte1_pcm.legs) if leg in join_legs1]
                joined1 = pte1_pcm.h[:, join_idxs]
                joined2 = pte2_pcm.h[:, join_idxs2]

                stacked = np.vstack([joined1 & 1, joined2 & 1])  
                r_join_stack = gf2_rank(stacked)
                    
                    
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
                    tn.ptes[node] = pte
                tn.legs_left_to_join[node_idx1] = [
                    leg
                    for leg in tn.legs_left_to_join[node_idx1]
                    if leg not in join_legs1
                ]
                tn.legs_left_to_join[node_idx2] = [
                    leg
                    for leg in tn.legs_left_to_join[node_idx2]
                    if leg not in join_legs2
                ]
                intermediate_tensor_sizes.append(len(pte.tensor))
                #estimated_cost =  (estimated_lengths[node_idx1] * estimated_lengths[node_idx2]) / 4
                keys_skipped = (2**(rank_submatrix)) / (2**(rank_submatrix1) * 2**(rank_submatrix2))
                estimated_cost = (2**(rank_submatrix1) * 2**(rank_submatrix2)) * keys_skipped
                if(r_join_stack != 2):
                    print("NOT EQUAL TO 2 during merge!, r_join_stack is: ", r_join_stack)
                if((2**(rank_submatrix1 + rank_submatrix2 - r_join_stack)) != ops_count):
                    num_mismatches_merge += 1
                    print("\t RANK != COSt")
                    print("MERGING PTEs: ", node_idx1, node_idx2)
                    print("\tnew pte length: ", len(pte.tensor), " full elements would be: ", 4**(open_legs1_est + open_legs2_est))
                    
                    # print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
                    # print("\t open legs1: ", open_legs1, " open legs 2: ", open_legs2)
                    # print("\t  merged pte, new length: ", len(pte.tensor))
                    print("\t  original lengths: ", len(node1_pte.tensor), len(node2_pte.tensor))
                    #print("\t estimated original lengths: ", estimated_lengths[node_idx1], estimated_lengths[node_idx2])
                    # #print("\t  merged pte parity check matrix: ", pte.h)
                    print("\t  cost: ", ops_count)
                        
                        
                    # print("\t------------------------")
                    print("\t submatrix rank: ", rank_submatrix)
                    print("\t rank of node 1: ", rank_submatrix1)
                    print("\t rank of node 2: ", rank_submatrix2)
                    print("\t rank of joined stacked: ", r_join_stack)

                    # print("\t parity check of node1: ", pte1_pcm.h)
                    # print("\t tracable legs of node1: ", node1_pte.tracable_legs)
                    # print("\t legs of node1: ", pte1_pcm.legs)
                    # print("\t join legs 1: ", join_legs1)
                    # print("\t parity check of node2: ", pte2_pcm.h)
                    # print("\t tracable legs of node2: ", node2_pte.tracable_legs)
                    # print("\t legs of node2: ", pte2_pcm.legs)
                    # print("\t join legs 2: ", join_legs2)
                    

                for node in merged_nodes_pcm:
                    estimated_lengths[node] = 2**rank_submatrix
                # print("\t conjoined parity check matrix: ")
                # print(np.array(new_pcm_pte.h))
                # print("\t\t pcm size: ", new_pcm_pte.h.shape)   
                # print("\t legs of the stabilizer tensor enumerator: ", new_pcm_pte.legs)
                # print("\t number of non mixed columns: ", count_non_mixed_columns(new_pcm_pte.h))
                # print("\t node 1 parity check matrix: ")
                # print(pte1_pcm.h)
                # print("\t node 1 shape: ", pte1_pcm.h.shape)
                # print("\t node 1 legs: ", pte1_pcm.legs)
                # print("\t node 2 parity check matrix: ")
                # print(pte2_pcm.h)
                # print("\t node 2 shape: ", pte2_pcm.h.shape)
                # print("\t node 2 legs: ", pte2_pcm.legs)
                # print("")

                update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : open_legs1_est, "open_legs2" : open_legs2_est, "pte1 sparsity" : len(node1_pte.tensor) / (4**open_legs1_est), "pte2 sparsity" : len(node2_pte.tensor) / (4**open_legs2_est)}, num_run, name, filename)

            total_ops_count += ops_count
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]

            for k in list(node1_pte.tensor.keys()):
                v = node1_pte.tensor[k]
                if tn.truncate_length is None:
                    continue
                if v.minw()[0] > tn.truncate_length:
                    del pte.tensor[k]
                else:
                    pte.tensor[k].truncate_inplace(tn.truncate_length)

        if len(set(tn.ptes.values())) > 1:
            pte_list = list(set(tn.ptes.values()))
            pte = pte_list[0]
            for pte2 in pte_list[1:]:
                pte, ops1, ops2 = pte.tensor_product(pte2, verbose=verbose)

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, score_cotengra, total_submatrix_rank_cost, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge


def get_contraction_time(tn, cotengra, num_run, name, filename):
    tn.analyze_traces(cotengra=cotengra, on_trial_error='raise', max_repeats=300)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, score_cotengra, total_submatrix_rank_cost, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge = stabilizer_enumerator_polynomial(
        tn, filename, num_run, name, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=cotengra
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, score_cotengra, total_submatrix_rank_cost, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge


def test_conjoining_cost(tn, traces, num_run, name, filename):
    # If there's only one node and no traces, return it directly
    if len(tn.nodes) == 1 and len(tn.traces) == 0:
        return list(tn.nodes.values())[0]
    
    rows = []
    cols = []
    total_pcm_cost = 0
    total_ops = 0
    total_legs_cost = 0
    total_rank_cost = 0
    ranks = []
    new_legs_cost = 0
    leg_cost_list = []
    total_non_mixed = 0
    total_submatrix_rank_cost = 0
    subrank_density = 0

    
    pte_lengths = {}
    open_legs = {}
    traceable_legs = {}
    for node_idx, pte in tn.ptes.items():
        pte_lengths[node_idx] = len(pte.tensor)
        open_legs[node_idx] = len(pte.tracable_legs) - 1
        traceable_legs[node_idx] = pte.tracable_legs

    print("Pte lengths to start are: ", pte_lengths)

    # Map from node_idx to the index of its PTE in ptes list
    nodes = list(tn.nodes.values())
    ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
        (node, {node.idx}) for node in nodes
    ]
    node_to_pte = {node.idx: i for i, node in enumerate(nodes)}

    for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
        pte1_len = pte_lengths.get(node_idx1, 0)
        pte2_len = pte_lengths.get(node_idx2, 0)
        print("Node indices: ", node_idx1, node_idx2)
        print("PTE lengths: ", pte1_len, pte2_len)

        join_legs1 = _index_legs(node_idx1, join_legs1)
        join_legs2 = _index_legs(node_idx2, join_legs2)

        pte1_idx = node_to_pte.get(node_idx1)
        pte2_idx = node_to_pte.get(node_idx2)

        open_legs1 = open_legs.get(node_idx1, 0)
        open_legs2 = open_legs.get(node_idx2, 0)


        # Case 1: Both nodes are in the same PTE
        if pte1_idx == pte2_idx:
            custom_cost = pte1_len * 0.25

            new_length = custom_cost
            pte_lengths[node_idx1] = new_length 
            pte_lengths[node_idx2] = new_length

            pte, nodes = ptes[pte1_idx]

            for node in nodes:
                pte_lengths[node] = new_length

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

            # print("tracable legs: ", traceable_legs[node_idx1])
            # print("New tensor length:")
            # print(compute_tensor_length(new_pte.h, traceable_legs[node_idx1]))

            rows.append(len(new_pte.h))
            cols.append(len(new_pte.h[0]))

            pcm_size_cost = 0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44
            rank = compute_gf2_rank(new_pte.h)
            ranks.append(rank)
            rank_cost = 2**(len(new_pte.legs) - rank)
            total_legs_cost += len(pte.legs) * 1.5
            leg_cost_list.append(len(pte.legs) * 1.5)

            open_legs_set = set(new_traceable_legs)
            open_leg_indices = [i for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_indices += [i + (new_pte.h.shape[1]//2) for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = gf2_rank(open_leg_submatrix)
            #print("new traceable legs after self trace: ", new_traceable_legs)

            prev_legs_set = set(new_traceable_legs + join_legs1 + join_legs2)
            prev_open_leg_indices = [i for i, leg in enumerate(pte.legs) if leg in prev_legs_set]
            prev_open_leg_indices += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in prev_legs_set]
            prev_open_leg_submatrix = pte.h[:, prev_open_leg_indices]
            prev_rank_submatrix = gf2_rank(prev_open_leg_submatrix)

            join_idxs = [i for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs2 = [i for i, leg in enumerate(pte.legs) if leg in join_legs1]
            join_idxs2 += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs1]
            joined1 = pte.h[:, join_idxs + join_idxs2]
            r_join_stack = gf2_rank(joined1)
            
            difference = r_join_stack - 2
            if(difference < 0):
                difference = 0
            subrank_density += 2**(prev_rank_submatrix - difference)

            # keys_skipped = (2**(rank_submatrix)) / (2**(prev_rank_submatrix))
            # subrank_density += (2**(prev_rank_submatrix))* keys_skipped

            # print("self tracing pte: ", pte)
            # print("\t nodes: ", nodes)
            # print("\t submatrix rank: ", rank_submatrix)
            # print("\t previous submatrix rank: ", prev_rank_submatrix)
            new_legs_cost += open_legs1 * 1.5

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, name, node_idx1, node_idx2, True, pte.h.shape, round(np.count_nonzero(pte.h) / pte.h.size, 3), -1, -1, new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3), pte1_len, -1, custom_cost, pcm_size_cost, rank_cost, rank, len(pte.legs), len(pte.legs), open_legs1, open_legs2, rank_submatrix, 2**(prev_rank_submatrix - (r_join_stack // 2))])

        # Case 2: Nodes are in different PTEs - merge them
        else:
            custom_cost = 0.25 * pte1_len * pte2_len
            new_length = custom_cost

            pte_lengths[node_idx1] = new_length
            pte_lengths[node_idx2] = new_length

            open_legs[node_idx1] = open_legs1 + open_legs2 - 1


            pte1, nodes1 = ptes[pte1_idx]
            pte2, nodes2 = ptes[pte2_idx]

            new_pte = pte1.conjoin(pte2, legs1=join_legs1, legs2=join_legs2)
            merged_nodes = nodes1.union(nodes2)
            # Update the first PTE with merged result
            ptes[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            ptes.pop(pte2_idx)

            total_legs_cost += len(pte1.legs) + len(pte2.legs) + np.min([len(pte1.legs), len(pte2.legs)])
            leg_cost_list.append(len(pte1.legs) + len(pte2.legs) + np.min([len(pte1.legs), len(pte2.legs)]))

            # print("tracable legs: ", traceable_legs[node_idx1] + traceable_legs[node_idx2])
            # print("New tensor length after merge:")
            # print(compute_tensor_length(new_pte.h, traceable_legs[node_idx1] + traceable_legs[node_idx2]))

            rows.append(len(new_pte.h))
            cols.append(len(new_pte.h[0]))

            for node in merged_nodes:
                pte_lengths[node] = new_length

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

            
            open_legs_set = set(new_traceable_legs)
            open_leg_indices = [i for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_indices += [i + (new_pte.h.shape[1]//2) for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            #print("Open legs indices: ", open_leg_indices)
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = gf2_rank(open_leg_submatrix)

            open_legs_set1 = set(traceable_legs[node_idx1])
            open_leg_indices1 = [i for i, leg in enumerate(pte1.legs) if leg in open_legs_set1]
            open_leg_indices1 += [i + (pte1.h.shape[1]//2) for i, leg in enumerate(pte1.legs) if leg in open_legs_set1]
            #print("Open legs indices: ", open_leg_indices)
            open_leg_submatrix1 = pte1.h[:, open_leg_indices1]
            rank_submatrix1 = gf2_rank(open_leg_submatrix1)

            open_legs_set2 = set(traceable_legs[node_idx2])
            open_leg_indices2 = [i for i, leg in enumerate(pte2.legs) if leg in open_legs_set2]
            open_leg_indices2 += [i + (pte2.h.shape[1]//2) for i, leg in enumerate(pte2.legs) if leg in open_legs_set2]
            #print("Open legs indices: ", open_leg_indices)
            open_leg_submatrix2 = pte2.h[:, open_leg_indices2]
            rank_submatrix2 = gf2_rank(open_leg_submatrix2)

            # keys_skipped = (2**(rank_submatrix)) / (2**(rank_submatrix_1) * 2**(rank_submatrix2))
            # subrank_density += (2**(rank_submatrix_1) * 2**(rank_submatrix2)) * keys_skipped
            

            join_idxs2 = [i for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs2 += [i + (pte2.h.shape[1]//2) for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs = [i for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            join_idxs += [i + (pte1.h.shape[1]//2) for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            joined1 = pte1.h[:, join_idxs]
            joined2 = pte2.h[:, join_idxs2]
            stacked = np.vstack([joined1 & 1, joined2 & 1])  
            r_join_stack = gf2_rank(stacked)

            subrank_density += (2**(rank_submatrix1 + rank_submatrix2 - r_join_stack))

            stacked = np.vstack([joined1 & 1, joined2 & 1])  
            r_join_stack = gf2_rank(stacked)
                    
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

            pcm_size_cost = 0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44
            
            #print("\t pte legs: ", new_pte.legs, " length: ", len(new_pte.legs))
            rank = compute_gf2_rank(new_pte.h)
            ranks.append(rank)
            rank_cost = 2**(len(new_pte.legs) - rank)
            # print("merging ptes: ", pte1, pte2)
            # print("\t nodes 1: ", nodes1, " nodes 2: ", nodes2)
            # print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
            # # print("New traceable legs after merge: ", new_traceable_legs)
            # # print("\t open legs1: ", open_legs1, " open legs 2: ", open_legs2)
            # # print("\t new legs: ", new_pte.legs)
            # # print("\t new pte pcm: ", new_pte.h)
            
            # print("submatrix rank after merge: ", rank_submatrix)
            # print("rank of node 1: ", rank_submatrix_1)
            # print("rank of node 2: ", rank_submatrix2)
            # print("\t\t pcm size: ", new_pte.h.shape)
            # print("\t original pte pcm 1: ", pte1.h)
            # print("\t original pte pcm 2: ", pte2.h)
            new_legs_cost += open_legs1 + open_legs2 + np.min([open_legs1, open_legs2])
            
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, name, node_idx1, node_idx2, False, pte1.h.shape, round(np.count_nonzero(pte1.h) / pte1.h.size, 3), pte2.h.shape, round(np.count_nonzero(pte2.h) / pte2.h.size, 3), new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3), pte1_len, pte2_len, custom_cost, pcm_size_cost, rank_cost, rank, len(pte1.legs), len(pte2.legs), open_legs1, open_legs2, rank_submatrix, (2**(rank_submatrix1 + rank_submatrix2 - r_join_stack))])

        total_pcm_cost += pcm_size_cost
        total_ops += custom_cost
        total_rank_cost += rank_cost
        total_non_mixed += open_legs1 + open_legs2 + np.min([open_legs1, open_legs2])
        total_submatrix_rank_cost += 2**rank_submatrix
        
    # If we have multiple components at the end, tensor them together
    if len(ptes) > 1:
        for other in ptes[1:]:
            ptes[0] = (ptes[0][0].tensor_with(other[0]), ptes[0][1].union(other[1]))

    return ptes[0][0], np.mean(rows), np.mean(cols), math.log2(total_pcm_cost), math.log2(total_ops), total_legs_cost, total_rank_cost, np.mean(rank), new_legs_cost, np.mean(leg_cost_list), total_submatrix_rank_cost, subrank_density


def run_wep_calc(q_shor, num_runs, coloring, distance, file_name):
    compass_code = CompassCode(distance, coloring)
    for name, rep in compass_code.get_representations().items():
        for i in range(num_runs):
            tn = rep()
            contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, score_cotengra, total_submatrix_rank_cost, subrank_w_self_trace, num_mismatches_self, num_mismatches_merge = get_contraction_time(tn, True, i, name, file_name)

            with open("outputs/results_7_25/new_submatrix_rank_calc.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([i, name, distance, contraction_time, total_ops_count, score_cotengra, contraction_width, total_submatrix_rank_cost, subrank_w_self_trace, num_mismatches_merge, num_mismatches_self])




filename = 'outputs/results_7_25/steps_for_new_self_trace.csv'

# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     writer.writerow(["num_run", "representation", "node_idx1", "node_idx2", "self trace?",
#                      "pcm_1 size", "pcm_1 sparsity", "pcm_2 size", "pcm_2 sparsity", "new pcm size", "new pcm sparsity",
#                      "pte 1 len estimate", "pte 2 len estimate",  "custom_cost",  "pcm size cost", "rank cost", "rank", "num pte legs1", "num pte legs2", 
#                      "open_legs1 est", "open_legs2 est", "rank_submatrix", "new_subrank_cost",
#                      "pte 1 len", "pte 2 len", "new pte len", "cost", "open_legs1", "open_legs2", 'pte1 sparsity', 'pte2 sparsity'])


# with open("new_submatrix_rank_calc.csv", 'w', newline='') as csvfile:
#      writer = csv.writer(csvfile, delimiter=';')
#      writer.writerow(["num_run", "representation", "distance", "contraction_time", "total_ops_count", "score cotengra", "contraction width", "total_submatrix_rank_cost", "subrank_w_self_trace", "num_mismatch_merge", "num_mismtach_self"])


def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


ds = [3,4,5]
q_shors = [0.0]

for d in ds:
    for q_shor in q_shors:
        coloring = generate_checkerboard_coloring(d)
        for i in range(d - 1):
            for j in range(d - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2

        run_wep_calc(q_shor, 1, coloring, d, filename)