import ast
import csv

from collections import defaultdict
import math
import re
import time
from typing import List, Set, Tuple
from galois import GF2
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
                print("Optimal ? traces from cotengra:")
                print(traces)
            contraction_width = tree.contraction_width()
            contraction_cost = tree.contraction_cost()
            score_cotengra = tree.get_score()
            flops, write = test_custom(tn, tree)
            pte_size_cost = test_custom2(tn, traces)
            final_pcm, estimated_ops, estimated_cost, total_open_legs_cost, combined_cost = test_conjoining_cost(tn, traces)
            print("tree default objective: ", tree.get_default_objective())
            #estimate_from_pcm = estimate_wep_from_pcm(final_pcm)
            #print("Final pcm shape: ", final_pcm.h.shape)

        summed_legs = [leg for leg in free_legs if leg not in open_legs]

        if len(tn.traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        total_ops_count = 0
        total_ops_count2 = 0
        open_legs_cost_actual = 0
        full_flops = 0
        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn.ptes else tn.ptes[node_idx2]
            #print("in Contraction: Node indices:", node_idx1, node_idx2, "Join legs:", join_legs1, join_legs2)
            #print("\t Node 1 PTE:", node1_pte, "Node 2 PTE:", node2_pte)
            # print("current node indices open legs:")
            # for node_idx, pte in tn.ptes.items():
            #     print(f"\tNode {node_idx}: {len(pte.tracable_legs)} open legs")
                
            # print("Node indices:", node_idx1, node_idx2)
            
            if node1_pte == node2_pte:
                # both nodes are in the same PTE!
                pte, ops_count, ops_count2, flops, open_legs = node1_pte.self_trace(
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
                #print("\t self tracing PTE")
                #print("\t \t open legs: ", open_legs)
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
                open_legs_cost_actual += 3*open_legs
            else:
                pte, ops_count, ops_count2, flops, open_legs1, open_legs2 = node1_pte.merge_with(
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
                
                #print("\t merging PTEs:")
                #print("\t \t open legs 1: ", open_legs1, " open legs 2: ", open_legs2)
                # print("\t\t sizes: ", len(node1_pte.tensor), len(node2_pte.tensor))
                # print("\t\t cost: ", ops_count)
                # print("\t\t new pte length: ", len(pte.tensor))
                
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
                open_legs_cost_actual += open_legs1 + open_legs2 + min(open_legs1, open_legs2)

            total_ops_count += ops_count
            total_ops_count2 += ops_count2
            full_flops += flops

            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]

            for k in list(node1_pte.tensor.keys()):
                v = node1_pte.tensor[k]
                if tn.truncate_length is None:
                    continue
                if v.minw()[0] > tn.truncate_length:
                    del pte.tensor[k]
                else:
                    pte.tensor[k].truncate_inplace(tn.truncate_length)
        print("Total operations after contraction: ", total_ops_count)
        if len(set(tn.ptes.values())) > 1:
            pte_list = list(set(tn.ptes.values()))
            pte = pte_list[0]
            print("End of contraction, still need to tensor product: ", len(pte_list[1:]), "PTEs")
            for pte2 in pte_list[1:]:
                pte, ops1, ops2, flops = pte.tensor_product(pte2, verbose=verbose)

                total_ops_count += ops1
                total_ops_count2 += ops2
                full_flops += flops

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, flops, write, pte_size_cost, estimated_ops, estimated_cost, total_open_legs_cost, open_legs_cost_actual, combined_cost


def get_contraction_time(tn, cotengra, max_repeats):
    tn.analyze_traces(cotengra=cotengra, on_trial_error='raise', max_repeats=max_repeats)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, flops, write, pte_size_cost, estimated_ops, avg_size_cost, open_legs_cost, open_legs_cost_actual, combined_cost = stabilizer_enumerator_polynomial(
        tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=cotengra
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra, flops, write, pte_size_cost, estimated_ops, avg_size_cost, open_legs_cost, open_legs_cost_actual, combined_cost

# def generate_generator_matrix(H):
#     # Compute generator matrix G from parity-check matrix H
#     # Over GF(2), so reduce mod 2
#     H = H % 2
#     # Get null space (mod 2): scipy gives real-valued, so round
#     G = null_space(H)
#     G = np.round(G) % 2
#     return G.astype(int)

# def estimate_weight_enumerator_terms(H, num_samples=10000):
#     G = generate_generator_matrix(H)
#     k = G.shape[0]
#     n = G.shape[1]

#     weights = set()
#     for _ in range(num_samples):
#         message = np.random.randint(0, 2, size=k)
#         codeword = np.dot(message, G) % 2
#         wt = np.sum(codeword)
#         weights.add(wt)
    
#     return sorted(weights), len(weights)

# def estimate_wep_from_pcm(pcm):
#     weights, len = estimate_weight_enumerator_terms(pcm.h)
#     return len

def test_custom2(tn, traces):
    ops_cost = 0
    seen = {}
    factor = 16
    
    for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
        multiplier = seen.get(node_idx1, 0)*factor + seen.get(node_idx2, 0)*factor
        pte1 = tn.ptes[node_idx1]
        pte2 = tn.ptes[node_idx2]
        
        if(multiplier == 0):
            multiplier = 1

        if(pte1 != pte2 and len(pte1.tensor) < 5000):
            new_cost = 0.25 * len(pte1.tensor) * len(pte2.tensor) * multiplier * len(pte1.tracable_legs) * len(pte2.tracable_legs)
        else:
            new_cost = len(pte1.tensor) * 0.25

        ops_cost += new_cost
        seen[node_idx1] = seen.get(node_idx1, 0) + 1
        seen[node_idx2] = seen.get(node_idx2, 0) + 1
    return math.log2(ops_cost)

def test_conjoining_cost(tn, traces):
    total_ops = 0
    total_pcm_cost = 0
    total_open_legs_cost = 0
    combined_cost = 0
    rows = []
    cols = []
    # If there's only one node and no traces, return it directly
    if len(tn.nodes) == 1 and len(tn.traces) == 0:
        return total_ops
    
    pte_lengths = {}
    open_legs = {}
    for node_idx, pte in tn.ptes.items():
        pte_lengths[node_idx] = len(pte.tensor)
        open_legs[node_idx] = len(pte.tracable_legs) - 1

    #print("Open legs for nodes to start:")
    #print(open_legs)


    # Map from node_idx to the index of its PTE in ptes list
    nodes = list(tn.nodes.values())
    ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
        (node, {node.idx}) for node in nodes
    ]
    node_to_pte = {node.idx: i for i, node in enumerate(nodes)}

    for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
        pte1_len = pte_lengths.get(node_idx1, 0)
        pte2_len = pte_lengths.get(node_idx2, 0)

        open_legs1 = open_legs.get(node_idx1, 0)
        open_legs2 = open_legs.get(node_idx2, 0)
        #print("Conjoining nodes:", node_idx1, node_idx2)
        #print("\t open legs 1:", open_legs1, " open legs 2:", open_legs2)

        join_legs1 = _index_legs(node_idx1, join_legs1)
        join_legs2 = _index_legs(node_idx2, join_legs2)

        pte1_idx = node_to_pte.get(node_idx1)
        pte2_idx = node_to_pte.get(node_idx2)


        # Case 1: Both nodes are in the same PTE
        if pte1_idx == pte2_idx:
            custom_cost = pte1_len * 0.25
            pte_lengths[node_idx1] = custom_cost * 0.5 # might need to change -- sometimes its 0.5 or / 8
            pte_lengths[node_idx2] = custom_cost * 0.5

            pte, nodes = ptes[pte1_idx]
            
            new_pte = pte.self_trace(join_legs1, join_legs2)
            ptes[pte1_idx] = (new_pte, nodes)

            rows.append(len(new_pte.h))
            cols.append(len(new_pte.h[0]))

            for node in nodes:
                open_legs[node] = open_legs1 - 2

            open_legs1 -= 1
            open_legs2 -= 1

            total_open_legs_cost += 0.384 * (open_legs1 + open_legs2) ** 2.47
            combined_cost += 4.463 * (open_legs1 * (0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44)) ** 0.46
            '''print("self tracing pte: ", pte)
            print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
            print("\t new pte pcm: ", new_pte.h)
            print("\t\t pcm size: ", new_pte.h.shape)
            print("\t original pte pcm: ", pte.h)'''

            '''with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, node_idx1, node_idx2, True, pte.h.shape, round(np.count_nonzero(pte.h) / pte.h.size, 3), -1, -1, new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3)])
'''
        # Case 2: Nodes are in different PTEs - merge them
        else:
            custom_cost = 0.25 * pte1_len * pte2_len
            pte_lengths[node_idx1] = custom_cost
            pte_lengths[node_idx2] = custom_cost

            open_legs[node_idx1] = open_legs1 + open_legs2 - 1


            pte1, nodes1 = ptes[pte1_idx]
            pte2, nodes2 = ptes[pte2_idx]

            new_pte = pte1.conjoin(pte2, legs1=join_legs1, legs2=join_legs2)
            merged_nodes = nodes1.union(nodes2)
            # Update the first PTE with merged result
            ptes[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            ptes.pop(pte2_idx)

            rows.append(len(new_pte.h))
            cols.append(len(new_pte.h[0]))

            for node in merged_nodes:
                open_legs[node] = open_legs1 + open_legs2 - 1

            # Update node_to_pte mappings
            for node in nodes2:
                node_to_pte[node] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node] = pte_idx - 1

            combined_cost += 4.463 * ((open_legs1 + open_legs2) * (0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44)) ** 0.46
            total_open_legs_cost += 0.384 * (open_legs1 + open_legs2 + min(open_legs1, open_legs2)) ** 2.47

            '''print("merging ptes: ", pte1, pte2)
            print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
            print("\t new pte pcm: ", new_pte.h)
            print("\t\t pcm size: ", new_pte.h.shape)
            print("\t original pte pcm 1: ", pte1.h)
            print("\t original pte pcm 2: ", pte2.h)'''

            '''with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, node_idx1, node_idx2, False, pte1.h.shape, round(np.count_nonzero(pte1.h) / pte1.h.size, 3), pte2.h.shape, round(np.count_nonzero(pte2.h) / pte2.h.size, 3), new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3)])
            '''
        total_ops += custom_cost
        total_pcm_cost += 0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44

        # Cost ≈ 3.354 * exp(0.081 * (m + n + min(m,n)))
        # total_open_legs_cost += 3.354 * math.exp(
        #     0.081 * (open_legs1 + open_legs2 + min(open_legs1, open_legs2))
        # )

            
        #print("updated open legs for nodes: ", open_legs)

    # If we have multiple components at the end, tensor them together
    if len(ptes) > 1:
        for other in ptes[1:]:
            ptes[0] = (ptes[0][0].tensor_with(other[0]), ptes[0][1].union(other[1]))

    
    return ptes[0][0], total_ops, total_pcm_cost, total_open_legs_cost, combined_cost


# def custom_cost_pcm_size_avg(tn, traces):
#     # If there's only one node and no traces, return it directly
#     if len(tn.nodes) == 1 and len(tn.traces) == 0:
#         return 0
#     # cost ≈ 0.00097 × (rows × cols)^2.44
#     total_cost = 0

#     # Map from node_idx to the index of its PTE in ptes list
#     nodes = list(tn.nodes.values())
#     ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
#         (node, {node.idx}) for node in nodes
#     ]
#     node_to_pte = {node.idx: i for i, node in enumerate(nodes)}

#     for node_idx1, node_idx2, join_legs1, join_legs2 in traces:
#         join_legs1 = _index_legs(node_idx1, join_legs1)
#         join_legs2 = _index_legs(node_idx2, join_legs2)

#         pte1_idx = node_to_pte.get(node_idx1)
#         pte2_idx = node_to_pte.get(node_idx2)


#         # Case 1: Both nodes are in the same PTE
#         if pte1_idx == pte2_idx:
#             pte, nodes = ptes[pte1_idx]
            
#             new_pte = pte.self_trace(join_legs1, join_legs2)
#             ptes[pte1_idx] = (new_pte, nodes)

#             cost = 0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44
            
#         # Case 2: Nodes are in different PTEs - merge them
#         else:
#             pte1, nodes1 = ptes[pte1_idx]
#             pte2, nodes2 = ptes[pte2_idx]

#             new_pte = pte1.conjoin(pte2, legs1=join_legs1, legs2=join_legs2)
#             merged_nodes = nodes1.union(nodes2)
#             # Update the first PTE with merged result
#             ptes[pte1_idx] = (new_pte, merged_nodes)
#             # Remove the second PTE
#             ptes.pop(pte2_idx)

#             cost = 0.00097 * (len(new_pte.h) * len(new_pte.h[0])) ** 2.44

#             # Update node_to_pte mappings
#             for node in nodes2:
#                 node_to_pte[node] = pte1_idx
#             # Adjust indices for all nodes in PTEs after the removed one
#             for node, pte_idx in node_to_pte.items():
#                 if pte_idx > pte2_idx:
#                     node_to_pte[node] = pte_idx - 1

#         total_cost += cost
#     # If we have multiple components at the end, tensor them together
#     if len(ptes) > 1:
#         for other in ptes[1:]:
#             ptes[0] = (ptes[0][0].tensor_with(other[0]), ptes[0][1].union(other[1]))

#     return total_cost

def test_custom(tn, tree):
    trial_dict = {'tree': tree}
    return custom(tn, trial_dict)


def custom(tn, trial_dict):
    # trial_dict: {'tree': <ContractionTree(N=18)>}
    tree = trial_dict['tree']
    trial_dict = tree.contract_stats()

    custom_cost = 0
    pte_sizes = 0
    '''for node, _, _ in tree.traverse():
        involved = tree.get_legs(node)
        for connection in involved:
            parts = connection.split('_')

            if(len(parts) == 2):
                # Parse each part into a Python tuple
                tuple1 = ast.literal_eval(parts[0])
                tuple2 = ast.literal_eval(parts[1])
                pte1 = tn.ptes[tuple1[0]]
                join_leg1 = tuple1[1]
                pte2 = tn.ptes[tuple2[0]]
                join_leg2 = tuple2[1]
                #pcm_cost += estimate_sparse_mult_cost(pte1.h, pte2.h)
                pte_sizes += len(pte1.tensor)*len(pte2.tensor)*0.25
                #custom_cost += estimate_merge_ops(pte1, pte2, [join_leg1], [join_leg2])
                #custom_cost += estimate_ops_count(pte1, pte2, [join_leg1], [join_leg2])
                node1 = tn.nodes[tuple1[0]]
                node2 = tn.nodes[tuple2[0]]
                custom_cost += estimate_join_cost(node1.h, node2.h, [join_leg1], [join_leg2])'''

    #score = math.log2(10 * trial_dict["flops"] + trial_dict["write"] + 10 * custom_cost)
    return trial_dict["flops"], trial_dict["write"]


#def get_wep_lengths_cost(pte1, pte2):



def estimate_ops_count(pte1, pte2, join_indices1, join_indices2):
    freq_pte2 = defaultdict(int)

    # Use GF2 + sslice to match join logic
    for key in pte2.tensor.keys():
        k2_gf2 = GF2(key)
        if any(i >= len(key)//2 for i in join_indices2):
            continue
        part = tuple(int(x) for x in sslice(k2_gf2, join_indices2))
        freq_pte2[part] += 1

    total_ops = 0
    for key in pte1.tensor.keys():
        k1_gf2 = GF2(key)
        if any(i >= len(key)//2 for i in join_indices1):
            continue
        part = tuple(int(x) for x in sslice(k1_gf2, join_indices1))
        total_ops += freq_pte2.get(part, 0)

    return total_ops





def estimate_sparse_mult_cost(h1, h2):
    # h1: (m x n), h2: (n x p)
    nnz1 = np.count_nonzero(h1)
    nnz2 = np.count_nonzero(h2)
    rows_b = h2.shape[0]
    if rows_b == 0:
        return float('inf')  # avoid divide-by-zero
    return nnz1 * (nnz2 / rows_b)


def estimate_join_cost(h1, h2, join_legs1, join_legs2):
    from collections import Counter

    # Project rows onto join legs
    join_view_1 = [tuple(int(x) for x in h1[i, join_legs1]) for i in range(h1.shape[0])]
    join_view_2 = [tuple(int(x) for x in h2[i, join_legs2]) for i in range(h2.shape[0])]

    # Count how many times each join pattern occurs in h2
    freq_2 = Counter(join_view_2)

    # For each row in h1, how many match in h2
    est_total = sum(freq_2[pattern] for pattern in join_view_1)

    return est_total


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
            if(distance > 4 and (name == "Measurement State Prep" or name == "Tanner Network")):
                continue
            tn = rep()
            contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, full_flops, score_cotengra,flops, write, pte_size_cost, estimated_ops, avg_size_cost, open_legs_cost, open_legs_cost_actual, combined_cost = get_contraction_time(tn, True, repeats)

            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([d, run_index, q_shor, repeats, name, round(contraction_time, 4), total_ops_count, total_ops_count2, full_flops, score_cotengra, contraction_cost, flops, write, pte_size_cost, estimated_ops, avg_size_cost, open_legs_cost, open_legs_cost_actual, combined_cost])    



#### Should think about how I want to run this in the future, wrap it all up into a nice function 
    ### with an option to name the output csv file as well

#with open('open_legs_tests/default_ctg_combined_cost.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile, delimiter=';')
#    writer.writerow(["distance", "run_index", "q_shor", "max_repeats", "representation", "contraction_time", "total_ops", "total_ops_mult", "total_ops_mult_add", "score_cotengra", "contraction_cost_cotengra", "cotengra_flops", "cotengra_write", "New PTE Calc", "Estimated Ops", "PCM Cost Model", "Open Legs Poly Cost", "Open Legs Count", "Combined Cost"])


num_runs = 100
ds = [5]
q_shors = [0.0]
max_repeats = [300]

for d in ds:
    for q_shor in q_shors:
        for repeats in max_repeats:
            run_wep_calc(q_shor, num_runs, d, 'open_legs_tests/default_ctg_combined_cost.csv', repeats)

'''with open('optimization_tests_with_default.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["distance", "contraction_time", "total_operations", "total_ops_w_length", "score_cotengra", "flops", "write", "New PCM cost", "PTE_tensor_sizes", "New PTE Calc"])

d = 4
coloring = generate_checkerboard_coloring(d)
compass = CompassCode(d, coloring)

for i in range(100):
    tn = compass.concatenated()

    contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, score_cotengra, score_custom, flops, write, pcm_cost, pte_sizes, pte_size_cost = get_contraction_time(tn, True)

    with open('optimization_tests_with_default.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([d, round(contraction_time, 3), total_ops_count, total_ops_count2, score_cotengra, flops, write, pcm_cost, pte_sizes, pte_size_cost])

'''