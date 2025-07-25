import ast
import csv

from collections import defaultdict
import random
import cotengra as ctg
import math
import re
import time
from typing import List, Set, Tuple
from galois import GF2
import galois
from scipy.linalg import null_space
import numpy as np
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


def stabilizer_enumerator_polynomial(
        tn,
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
            estimated_cost = test_conjoining_cost(tn, traces)
            print("tree default objective: ", tree.get_default_objective())

        summed_legs = [leg for leg in free_legs if leg not in open_legs]

        if len(tn.traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        total_ops_count = 0
        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn.ptes else tn.ptes[node_idx2]
            #print("in Contraction: Node indices:", node_idx1, node_idx2, "Join legs:", join_legs1, join_legs2)
            #print("\t Node 1 PTE:", node1_pte, "Node 2 PTE:", node2_pte)

            if node1_pte == node2_pte:
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
                # print("\t self tracing PTE")
                # print("\t\t sizes: ", len(node1_pte.tensor), len(node2_pte.tensor))
                # print("\t\t cost: ", ops_count)
                # print("\t\t new pte length: ", len(pte.tensor))
                # print("\t\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
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
                pte, ops1 = pte.tensor_product(pte2, verbose=verbose)
                total_ops_count += ops1

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, estimated_cost


def get_contraction_time(tn, cotengra, max_repeats):
    tn.analyze_traces(cotengra=cotengra, on_trial_error='raise', max_repeats=max_repeats)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, submatrix_rank_cost = stabilizer_enumerator_polynomial(
        tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=cotengra
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, submatrix_rank_cost


def compute_gf2_rank(pcm_sub):
    GF2 = galois.GF(2)
    pcm_gf2 = GF2(pcm_sub)
    return pcm_gf2.row_space().shape[0]


def test_conjoining_cost(tn, traces):
    # If there's only one node and no traces, return it directly
    if len(tn.nodes) == 1 and len(tn.traces) == 0:
        return list(tn.nodes.values())[0]
    
    total_submatrix_rank_cost = 0

    pte_lengths = {}
    open_legs = {}
    traceable_legs = {}
    for node_idx, pte in tn.ptes.items():
        pte_lengths[node_idx] = len(pte.tensor)
        open_legs[node_idx] = len(pte.tracable_legs) - 1
        traceable_legs[node_idx] = pte.tracable_legs

    # Map from node_idx to the index of its PTE in ptes list
    nodes = list(tn.nodes.values())
    ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
        (node, {node.idx}) for node in nodes
    ]
    node_to_pte = {node.idx: i for i, node in enumerate(nodes)}

    for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
        pte1_len = pte_lengths.get(node_idx1, 0)
        pte2_len = pte_lengths.get(node_idx2, 0)

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


            open_legs_set = set(new_traceable_legs)
            open_leg_indices = [i for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_indices += [i + (new_pte.h.shape[1]//2) for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = compute_gf2_rank(open_leg_submatrix)

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
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = compute_gf2_rank(open_leg_submatrix)

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

        total_submatrix_rank_cost += 2**rank_submatrix
        
    return total_submatrix_rank_cost



def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


def run_wep_calc(q_shor, num_runs, distance, file_name, repeats):
    for run_index in range(num_runs):
        # Generate a new coloring per run
        coloring = generate_checkerboard_coloring(distance)
        for i in range(distance - 1):
            for j in range(distance - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2

        compass_code = CompassCode(distance, coloring)

        for name, rep in compass_code.get_representations().items():
            # if(distance > 4 and (name == "Measurement State Prep" or name == "Tanner Network")):
            #     continue
            tn = rep()
            contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, submatrix_rank_cost = get_contraction_time(tn, True, repeats)

            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([d, run_index, q_shor, repeats, "size", name, round(contraction_time, 4), total_ops_count, score_cotengra, submatrix_rank_cost])    


# with open('optimal_cotengra.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     writer.writerow(["distance", "run_index", "q_shor", "max_repeats", "minimize", "representation", "contraction_time", "total_ops", "score_cotengra", "submatrix_rank_cost"])


num_runs = 1
ds = [3]
q_shors = [0.0]
#max_repeats = [512, 256, 128, 64, 16]
max_repeats = [300]

for d in ds:
    for q_shor in q_shors:
        for repeats in max_repeats:
            run_wep_calc(q_shor, num_runs, d, 'optimal_cotengra.csv', repeats)
