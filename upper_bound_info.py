import ast
import csv
import math

from collections import defaultdict
import time
from typing import List, Set, Tuple
import numpy as np
from qlego.linalg import gauss
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.stabilizer_tensor_enumerator import (
    StabilizerCodeTensorEnumerator,
    _index_leg,
    _index_legs,
)

import qlego
from qlego.parity_check import sprint
from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator

tensor_cache = {}

def run_upper_bound_calc(
        tn,
        open_legs: List[Tuple[int, int]] = [],
        verbose: bool = False,
        progress_reporter: ProgressReporter = DummyProgressReporter(),
        cotengra: bool = True,
    ):
        contraction_width = -1
        contraction_cost = -1
        result = 0

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

        # for node_idx, node in tn.nodes.items():
        #     #print("Node: ", node)
        #     traced_legs = open_legs_per_node[node_idx]  


        #     # open_legs_set = set(traced_legs)
        #     # open_leg_indices = [i for i, leg in enumerate(node.legs) if leg in open_legs_set]
        #     # open_leg_indices += [i + (node.h.shape[1]//2) for i, leg in enumerate(node.legs) if leg in open_legs_set]
        #     # open_leg_submatrix = node.h[:, open_leg_indices]
        #     open_leg_indices = [node.legs.index(leg) for leg in traced_legs]

        #     matrix_key = (tuple(tuple(int(x) for x in row) for row in gauss(node.h)), tuple(open_leg_indices))
        #     # print(node_idx, traced_legs)
        #     # sprint(gauss(node.h))

        #     # full parity check matrix + equal open legs

        #     if(matrix_key in tensor_cache):
        #         tensor = tensor_cache[matrix_key]
        #     else:
        #         tensor = node.stabilizer_enumerator_polynomial(
        #             open_legs=traced_legs,
        #             verbose=verbose,
        #             progress_reporter=progress_reporter,
        #             truncate_length=tn.truncate_length,
        #         )
        #         if(tensor in tensor_cache.values()):
        #             print("duplicate tensor found!!")
        #         tensor_cache[matrix_key] = tensor
                
        #     if len(traced_legs) == 0:
        #         tensor = {(): tensor}
        #     tn.ptes[node_idx] = _PartiallyTracedEnumerator(
        #         nodes={node_idx},
        #         tracable_legs=open_legs_per_node[node_idx],
        #         tensor=tensor,  
        #         truncate_length=tn.truncate_length,
        #     )

        if cotengra and len(tn.nodes) > 0 and len(tn.traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, verbose, progress_reporter
                )

        return find_upper_bound_cost(tn, traces, open_legs_per_node)
       

def find_upper_bound_cost(tn, traces, open_legs_per_node):
    total_legs_count = 0

    open_legs_count = {node_idx: len(legs) for node_idx, legs in open_legs_per_node.items()}

    nodes = list(tn.nodes.values())
    ptes: List[Tuple[StabilizerCodeTensorEnumerator, Set[str]]] = [
        (node, {node.idx}) for node in nodes
    ]
    node_to_pte = {node.idx: i for i, node in enumerate(nodes)}

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

            for node in nodes:
                open_legs_count[node] = open_legs1 - 2

            open_legs1 -= 1
            open_legs2 -= 1
            
            new_pte = pte.self_trace(join_legs1, join_legs2)
            ptes[pte1_idx] = (new_pte, nodes)

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

            for node in merged_nodes:
                open_legs_count[node] = open_legs1 + open_legs2 - 1

            # Update node_to_pte mappings
            for node in nodes2:
                node_to_pte[node] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node] = pte_idx - 1

        exp = int(open_legs1) + int(open_legs2) + min(int(open_legs1), int(open_legs2))
        total_legs_count += 2 ** exp

    return math.log2(total_legs_count)


def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


with open("upper_bound_results.csv", "w") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["distance", "representation", "num_run", "upper_bound_cost"])

ds = [3, 5, 7, 9, 11, 13, 15, 17]
num_runs = 100

for distance in ds:
    coloring = generate_checkerboard_coloring(distance)
    compass_code = CompassCode(distance, coloring)

    for name, rep in compass_code.get_representations().items():
        costs = []

        for i in range(num_runs):  
            tn = rep()
            cost = run_upper_bound_calc(tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)
            
            with open(f"upper_bound_results.csv", "a") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([distance, name, i, cost])
