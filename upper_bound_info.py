import ast
import csv
import itertools
import math
import sys
import os

from compassCodes.holographic import HolographicHappyTN

from bb_parity_check import create_coprime_parity_check, create_full_parity_check
from planqtn.networks.compass_code import CompassCodeRotatedRectangularTN
from planqtn.networks.stabilizer_measurement_state_prep import StabilizerMeasurementStatePrepTN

from galois import GF2
from planqtn.linalg import rank
from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple
from planqtn.parity_check import tensor_product
from planqtn.progress_reporter import TqdmProgressReporter
from planqtn.progress_reporter import DummyProgressReporter, ProgressReporter
from planqtn.stabilizer_tensor_enumerator import (
    StabilizerCodeTensorEnumerator,
    _index_leg,
    _index_legs,
)

from compassCodes.compass_code import CompassCode

tensor_cache = {}

def run_upper_bound_calc(
        tn,
        open_legs: List[Tuple[int, int]] = [],
        verbose: bool = False,
        progress_reporter: ProgressReporter = DummyProgressReporter(),
        cotengra: bool = True,
    ):

        if tn._wep is not None:
            return tn._wep

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


        if cotengra and len(tn.nodes) > 0 and len(tn._traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, open_legs_per_node, verbose, progress_reporter
                )

        return find_upper_bound_cost(tn, traces, open_legs_per_node)
       

def matching_stabilizers_merge(g1, g2):
    tensor_prod = tensor_product(g1, g2)

    A = GF2([[1, 1, 0, 0],
             [0, 0, 1, 1]])

    projection = tensor_prod @ A.T
    return rank(projection)


def fast_matching_stabilizer_ratio(generators):
    """
    Compute the ratio of stabilizers that act identically on the qubits
    (i.e., XX, YY, ZZ, or II), using linear algebra over GF(2).
    """

    parity_constraints = GF2([
        [1, 1, 0, 0],  # x0 + x1 = 0
        [0, 0, 1, 1]   # z0 + z1 = 0
    ])

    
    parity_vector = GF2(generators) @ parity_constraints.T 
    return rank(parity_vector)

def find_upper_bound_cost(tn, traces, open_legs_per_node):
    total_legs_count = 0
    custom_cost_total = 0

    open_legs_count = {node_idx: len(legs) for node_idx, legs in open_legs_per_node.items()}
    traceable_legs = {}
    for node_idx, legs in open_legs_per_node.items():
        traceable_legs[node_idx] = legs
        
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

        open_legs1 = open_legs_count.get(node_idx1, 0)
        open_legs2 = open_legs_count.get(node_idx2, 0)


        # Case 1: Both nodes are in the same PTE
        if pte1_idx == pte2_idx:
            pte, nodes = ptes[pte1_idx]

            new_traceable_legs = [
                leg
                for leg in traceable_legs[node_idx1]
                if leg not in join_legs1 and leg not in join_legs2
            ]

            for node in nodes:
                open_legs_count[node] = open_legs1 - 2
                traceable_legs[node] = new_traceable_legs

            open_legs1 -= 1
            open_legs2 -= 1
            
            new_pte = pte.self_trace(join_legs1, join_legs2)
            ptes[pte1_idx] = (new_pte, nodes)

            prev_rank_submatrix = tn._get_rank_for_matrix_legs(pte, new_traceable_legs + join_legs1 + join_legs2)

            join_idxs = [i for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs2]
            join_idxs2 = [i for i, leg in enumerate(pte.legs) if leg in join_legs1]
            join_idxs2 += [i + (pte.h.shape[1]//2) for i, leg in enumerate(pte.legs) if leg in join_legs1]
            joined1 = pte.h[:, [join_idxs[0], join_idxs2[0], join_idxs[1], join_idxs2[1]]]
            
            #matches = tn.count_matching_stabilizers_by_enumeration(joined1)
            #custom_cost_total += 2**(prev_rank_submatrix)* matches
            matches = fast_matching_stabilizer_ratio(joined1)
            custom_cost_total += 2**(prev_rank_submatrix - matches)

        # Case 2: Nodes are in different PTEs - merge them
        else:
            open_legs_count[node_idx1] = open_legs1 + open_legs2 - 1

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
            prev_submatrix1 = tn._get_rank_for_matrix_legs(pte1, traceable_legs[node_idx1])
            prev_submatrix2 = tn._get_rank_for_matrix_legs(pte2, traceable_legs[node_idx2])

            join_idxs2 = [i for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs2 += [i + (pte2.h.shape[1]//2) for i, leg in enumerate(pte2.legs) if leg in join_legs2]
            join_idxs = [i for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            join_idxs += [i + (pte1.h.shape[1]//2) for i, leg in enumerate(pte1.legs) if leg in join_legs1]
            joined1 = pte1.h[:, join_idxs]
            joined2 = pte2.h[:, join_idxs2]

            #tensor_prod = tensor_product(joined1, joined2)                
            #matches = tn.count_matching_stabilizers_by_enumeration(tensor_prod)
            #custom_cost_total += (2**(prev_submatrix1 + prev_submatrix2)* matches)
            matches = matching_stabilizers_merge(joined1, joined2)
            custom_cost_total += (2**(prev_submatrix1 + prev_submatrix2 - matches))

            for node in merged_nodes:
                open_legs_count[node] = open_legs1 + open_legs2 - 1
                traceable_legs[node] = new_traceable_legs

            # Update node_to_pte mappings
            for node in nodes2:
                node_to_pte[node] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node] = pte_idx - 1

        exp = int(open_legs1) + int(open_legs2) + min(int(open_legs1), int(open_legs2))
        total_legs_count += 2 ** exp

    return math.log(total_legs_count), math.log(custom_cost_total)





def generate_checkerboard_coloring(d, L=None):
    if(L is None):
        L = d
    return [[1 + (i + j) % 2 for j in range(L-1)] for i in range(d-1)]


def generate_hamming_hx(r):
    bitstrings = [np.array(bits) for bits in itertools.product([0, 1], repeat=r) if any(bits)]
    H = np.array(bitstrings).T
    return H

# with open("upper_bound_and_custom_with_custom_cost.csv", "w") as f:
#     writer = csv.writer(f, delimiter=';')
#     writer.writerow(["distance", "representation", "num_run", "upper_bound_cost", "custom_cost"])

# ds = [7, 9, 11, 13, 15, 17]
# num_runs = 2

# for distance in ds:
#     coloring = generate_checkerboard_coloring(distance)
#     compass_code = CompassCode(distance, coloring)

#     for name, rep in compass_code.get_representations().items():
#         costs = []

#         for i in range(num_runs):  
#             tn = rep()
#             legs_cost, custom_cost = run_upper_bound_calc(tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)
            
#             with open(f"upper_bound_and_custom_with_custom_cost.csv", "a") as f:
#                 writer = csv.writer(f, delimiter=';')
#                 writer.writerow([distance, name, i, legs_cost, custom_cost, 30])

# d = 3
# coloring = np.full((d, d), 2)
# compass_code = CompassCode(d, coloring)
# tn_concat = compass_code.concatenated()
# legs_cost, custom_cost = run_upper_bound_calc(tn_concat, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

# print("custom cost: ", custom_cost)

# with open("tn_architectures_costs.csv", "w") as f:
#      writer = csv.writer(f, delimiter=';')
#      writer.writerow(["distance", "num_qubits", "representation", "num_run", "upper_bound_cost", "custom_cost"])


ds = [15, 17]
num_runs = 10

for d in ds:
    for i in range(num_runs):
        #Symmetric Tree Tensor Network (Concatenated Shor's):
        compass_code_shor = CompassCode(d, np.full((d, d), 2))
        tn_concat = compass_code_shor.concatenated()
        
        legs_cost, custom_cost = run_upper_bound_calc(tn_concat, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([d, d**2, "Concatenated", i, legs_cost, custom_cost])


        #Rotated Surface Method
        compass_code_surface = CompassCode(d, generate_checkerboard_coloring(d))
        tn_surface = compass_code_surface.rotated()
    
        legs_cost, custom_cost = run_upper_bound_calc(tn_surface, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([d, d**2, "Rotated Surface", i, legs_cost, custom_cost])
    

        #Rectangular Rotated Surface
        coloring = generate_checkerboard_coloring(d, d+1)
        tn_rectangular = CompassCodeRotatedRectangularTN(coloring)
        legs_cost, custom_cost = run_upper_bound_calc(tn_rectangular, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([d, (d*(d+1)), "Rectangular Surface", i, legs_cost, custom_cost])

for d in [3, 5, 7]:
    for i in range(num_runs):
        #MSP for Hamming Code (non-degenerate)
        h_x_hamming = generate_hamming_hx(d)
        top = np.hstack((h_x_hamming, np.zeros([len(h_x_hamming), len(h_x_hamming[0])])))
        bottom = np.hstack((np.zeros([len(h_x_hamming), len(h_x_hamming[0])]), h_x_hamming))
        H_hamming = np.vstack((top, bottom))
        tn_hamming = StabilizerMeasurementStatePrepTN(H_hamming)

        legs_cost, custom_cost = run_upper_bound_calc(tn_hamming, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([d, (2**d - 1), "Hamming MSP (non-degenerate)", i, legs_cost, custom_cost])
    

layers = [5] 
for layer in layers:
    for i in range(num_runs):
        tn_holo = HolographicHappyTN(layer)
        print(tn_holo.nodes)
        legs_cost, custom_cost = run_upper_bound_calc(tn_holo, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([3, tn_holo.n, "Holographic", i, legs_cost, custom_cost])
    

# BB Code Params:

ls = [3, 3, 6, 12, 12]
ms = [3, 5, 6, 6, 12]
a_s = [[1, 0, 2], [0, 1, 2], [3, 1, 2], [3, 1, 2], [3, 2, 7]]
bs = [[1, 0, 2], [1, 3, 8], [3, 1, 2], [3, 1, 2], [3, 1, 2]]

# firs two are coprime BB Codes: 18 and 30 qubit
for i in range(2):
    for j in range(num_runs):
        H_coprime = create_coprime_parity_check(ls[i], ms[i], a_s[i], bs[i])
        tn_bb = StabilizerMeasurementStatePrepTN(H_coprime)
        legs_cost, custom_cost = run_upper_bound_calc(tn_bb, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([-1, tn_bb.n_qubits(), "BB MSP (degenerate)", j, legs_cost, custom_cost])
    

# nest ones are: 72, 144, 288
for i in range(2, len(ls)):
    for j in range(num_runs):
        H_reg = create_full_parity_check(ls[i], ms[i], a_s[i], bs[i])
        tn_bb = StabilizerMeasurementStatePrepTN(H_reg)

        legs_cost, custom_cost = run_upper_bound_calc(tn_bb, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

        with open(f"tn_architectures_costs.csv", "a") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([-1, tn_bb.n_qubits(), "BB MSP (degenerate)", j, legs_cost, custom_cost])
        
    

