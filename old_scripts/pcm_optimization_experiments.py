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
from sympy import Matrix

from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator
from qlego.symplectic import sslice

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
            
            final_pcm, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost = test_conjoining_cost(tn, traces, num_run, name, filename)

        summed_legs = [leg for leg in free_legs if leg not in open_legs]

        if len(tn.traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        total_ops_count = 0
        total_ops_count2 = 0
        print("GOING INTO ACTUAL CONTRACTION NOW")
        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn.ptes else tn.ptes[node_idx2]

            if node1_pte == node2_pte:
                #print("Self tracing pte: ", node1_pte)
                #print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
                # both nodes are in the same PTE!
                pte, ops_count, ops_count2, full_flops, open_legs = node1_pte.self_trace(
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
                # print("\t  self traced pte, new length: ", len(pte.tensor))
                # print("\t  original length: ", len(node1_pte.tensor))
                # #print("\t self traced pte parity check matrix: ", pte.h)
                # print("\t  cost: ", ops_count)
                # print("\t  open legs1: ", open_legs)
                # print("\t  pte sparsity: ", len(node1_pte.tensor) / (4**open_legs))
                update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : open_legs, "open_legs2": open_legs, "pte1 sparsity" : len(node1_pte.tensor) / (4**open_legs), "pte2 sparsity": len(node1_pte.tensor) / (4**open_legs)}, num_run, name, filename)

            else:
                pte, ops_count, ops_count2, full_flops, open_legs1, open_legs2, tracable_legs = node1_pte.merge_with(
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
                print("new pte length: ", len(pte.tensor), " full elements would be: ", 4**(open_legs1 + open_legs2))
                if(len(pte.tensor) < 4**(open_legs1 + open_legs2)):
                    print("merging ptes: ", node1_pte, node2_pte)
                    print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
                    print("\t  merged pte, new length: ", len(pte.tensor))
                    print("\t  original lengths: ", len(node1_pte.tensor), len(node2_pte.tensor))
                    #print("\t  merged pte parity check matrix: ", pte.h)
                    print("\t  cost: ", ops_count)
                    print("\t open legs1: ", open_legs1, " open legs 2: ", open_legs2)
                    print("\t open legs: ", pte.tracable_legs)
                    print("\t pte1 sparsity: ", len(node1_pte.tensor) / (4**open_legs1), " pte2 sparsity: ", len(node2_pte.tensor) / (4**open_legs2))
                update_csv_row_pandas(node_idx1, node_idx2, {"pte 1 len": len(node1_pte.tensor), "pte 2 len" : len(node2_pte.tensor), "new pte len": len(pte.tensor), "cost" : ops_count, "open_legs1" : open_legs1, "open_legs2" : open_legs2, "pte1 sparsity" : len(node1_pte.tensor) / (4**open_legs1), "pte2 sparsity" : len(node2_pte.tensor) / (4**open_legs2)}, num_run, name, filename)

            total_ops_count += ops_count
            total_ops_count2 += ops_count2
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

        total_ops_count += ops_count
        total_ops_count2 += ops_count2

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, score_cotengra, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost


def get_contraction_time(tn, cotengra, num_run, name, filename):
    tn.analyze_traces(cotengra=cotengra, on_trial_error='raise', max_repeats=300)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, score_cotengra, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost = stabilizer_enumerator_polynomial(
        tn, filename, num_run, name, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=cotengra
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, score_cotengra, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost

def compute_gf2_rank(pcm_sub):
    """
    pcm_sub: a numpy array of shape (m, n) with 0s and 1s
    Returns the rank over GF(2)
    """
    GF2 = galois.GF(2)
    pcm_gf2 = GF2(pcm_sub)
    return pcm_gf2.row_space().shape[0]


def count_non_mixed_columns(pcm):
    """
    Count the number of columns in the parity check matrix that are:
    - all zeros (no checks),
    - only involved in X-checks,
    - or only involved in Z-checks.
    
    pcm: numpy array of shape (m, n) where left half are X-checks and right half Z-checks.
    Returns: count of such columns.
    """
    m, n = pcm.shape
    mid = n // 2  # assuming left half is X and right half is Z

    count = 0
    for col in range(n):
        column_data = pcm[:, col]
        has_check = np.any(column_data)
        
        # Check if column has only X checks
        if col < mid:
            # Only X checks if column is nonzero and the corresponding Z column is zero
            if has_check and not np.any(pcm[:, col + mid]):
                count += 1
        # Check if column has only Z checks
        elif col >= mid:
            if has_check and not np.any(pcm[:, col - mid]):
                count += 1
        # Column has no checks
        if not has_check:
            count += 1

    return count


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

            # print("self tracing pte: ", pte)
            # print("\t nodes: ", nodes)
            # print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
            # print("\t new pte pcm: ", new_pte.h)
            # print("\t\t pcm size: ", new_pte.h.shape)
            # print("\t original pte pcm: ", pte.h)
            new_legs_cost += open_legs1 * 1.5
            num_non_mixed = count_non_mixed_columns(new_pte.h)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, name, node_idx1, node_idx2, True, pte.h.shape, round(np.count_nonzero(pte.h) / pte.h.size, 3), -1, -1, new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3), pte1_len, -1, custom_cost, pcm_size_cost, rank_cost, rank, len(pte.legs), len(pte.legs), open_legs1, open_legs2, rank_submatrix])

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
            print("Open legs indices: ", open_leg_indices)
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = gf2_rank(open_leg_submatrix)

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
            
            print("\t pte legs: ", new_pte.legs, " length: ", len(new_pte.legs))
            rank = compute_gf2_rank(new_pte.h)
            ranks.append(rank)
            rank_cost = 2**(len(new_pte.legs) - rank)
            print("merging ptes: ", pte1, pte2)
            print("\t nodes 1: ", nodes1, " nodes 2: ", nodes2)
            print("New traceable legs after merge: ", new_traceable_legs)
            print("open legs 1: ", open_legs1, " open legs 2: ", open_legs2)
            print("\t join legs 1: ", join_legs1, " join legs 2: ", join_legs2)
            print("\t new legs: ", new_pte.legs)
            print("\t new pte pcm: ", new_pte.h)
            
            print("\t\t pcm size: ", new_pte.h.shape)
            print("\t original pte pcm 1: ", pte1.h)
            print("\t original pte pcm 2: ", pte2.h)
            new_legs_cost += open_legs1 + open_legs2 + np.min([open_legs1, open_legs2])
            
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([num_run, name, node_idx1, node_idx2, False, pte1.h.shape, round(np.count_nonzero(pte1.h) / pte1.h.size, 3), pte2.h.shape, round(np.count_nonzero(pte2.h) / pte2.h.size, 3), new_pte.h.shape, round(np.count_nonzero(new_pte.h) / new_pte.h.size, 3), pte1_len, pte2_len, custom_cost, pcm_size_cost, rank_cost, rank, len(pte1.legs), len(pte2.legs), open_legs1, open_legs2, rank_submatrix])

        total_pcm_cost += pcm_size_cost
        total_ops += custom_cost
        total_rank_cost += rank_cost
        total_non_mixed += open_legs1 + open_legs2 + np.min([open_legs1, open_legs2])
        total_submatrix_rank_cost += 2**rank_submatrix
        
    # If we have multiple components at the end, tensor them together
    if len(ptes) > 1:
        for other in ptes[1:]:
            ptes[0] = (ptes[0][0].tensor_with(other[0]), ptes[0][1].union(other[1]))

    return ptes[0][0], np.mean(rows), np.mean(cols), math.log2(total_pcm_cost), math.log2(total_ops), total_legs_cost, total_rank_cost, np.mean(rank), new_legs_cost, np.mean(leg_cost_list), sum_top_x_percent(leg_cost_list, 0.05), sum_top_x_percent(leg_cost_list, 0.1), sum_top_x_percent(leg_cost_list, 0.2), sum_top_x_percent(leg_cost_list, 0.02), sum_top_x_percent(leg_cost_list, 0.01), total_non_mixed, total_submatrix_rank_cost

def sum_top_x_percent(arr, x):
    n = len(arr)
    count_top = math.ceil(x * n)
    print(count_top)
    top_values = sorted(arr, reverse=True)[:count_top]
    return sum(top_values)

def run_wep_calc(q_shor, num_runs, coloring, distance, file_name):
    compass_code = CompassCode(distance, coloring)
    for name, rep in compass_code.get_representations().items():
        total_ops = 0
        total_ops2 = 0
        durations = []

        for i in range(num_runs):
            tn = rep()
            contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, total_ops_count2, score_cotengra, avg_rows, avg_cols, pcm_cost, estimated_ops, total_legs_cost, total_rank_cost, avg_rank, new_legs_cost, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost = get_contraction_time(tn, True, i, name, file_name)
            durations.append(contraction_time)
            total_ops += total_ops_count
            total_ops2 += total_ops_count2

            with open("t.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([i, name, distance, contraction_time, total_ops_count, avg_rows, avg_cols, pcm_cost, estimated_ops, score_cotengra, total_legs_cost, new_legs_cost, total_rank_cost, contraction_width, avg_rank, avg_legs, top_5_legs, top_10_legs, top_20_legs, top_2_legs, top_1_legs, num_non_mixed, total_submatrix_rank_cost])




filename = 'test.csv'

# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=';')
#     writer.writerow(["num_run", "representation", "node_idx1", "node_idx2", "self trace?",
#                      "pcm_1 size", "pcm_1 sparsity", "pcm_2 size", "pcm_2 sparsity", "new pcm size", "new pcm sparsity",
#                      "pte 1 len estimate", "pte 2 len estimate",  "custom_cost",  "pcm size cost", "rank cost", "rank", "num pte legs1", "num pte legs2", "open_legs1 est", "open_legs2 est", "rank_submatrix",
#                      "pte 1 len", "pte 2 len", "new pte len", "cost", "open_legs1", "open_legs2", 'pte1 sparsity', 'pte2 sparsity'])


# with open("t.csv", 'w', newline='') as csvfile:
#      writer = csv.writer(csvfile, delimiter=';')
#      writer.writerow(["num_run", "representation", "distance", "contraction_time", "total_ops_count", "avg rows", "avg cols", "pcm cost", "pte len cost", "score cotengra", "total legs cost", "new legs cost", "rank cost", "contraction width", "avg rank", "avg_legs", "top_5_legs", "top_10_legs", "top_20_legs", "top_2_legs", "top_1_legs", "non_mixed_cols", "total_submatrix_rank_cost"])


def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


ds = [3]
q_shors = [0.0]

for d in ds:
    for q_shor in q_shors:
        coloring = generate_checkerboard_coloring(d)
        for i in range(d - 1):
            for j in range(d - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2

        run_wep_calc(q_shor, 1, coloring, d, filename)