import ast
import csv
import math
import os

import galois
import qlego
import sys
sys.path.insert(0, "/Users/junevanlerberghe/Documents/Duke/research/repositories/compass-wep-complexity/tnqec")


from collections import defaultdict
import time
from typing import List, Set, Tuple
import numpy as np
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.stabilizer_tensor_enumerator import (
    StabilizerCodeTensorEnumerator,
    _index_leg,
    _index_legs,
)

import qlego
from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator

def run_upper_bound_calc(
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

        if cotengra and len(tn.nodes) > 0 and len(tn.traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, verbose, progress_reporter
                )

        return custom_cost_submatrix_rank(tn, traces)


def compute_gf2_rank(self, pcm_sub):
    GF2 = galois.GF(2)
    pcm_gf2 = GF2(pcm_sub)
    return pcm_gf2.row_space().shape[0]


def custom_cost_submatrix_rank(tn, traces):
    # If there's only one node and no traces, return it directly
    if len(tn.nodes) == 1 and len(tn.traces) == 0:
        return 0

    total_cost = 0

    traceable_legs = {}
    for node_idx, pte in tn.ptes.items():
        traceable_legs[node_idx] = pte.tracable_legs

    # Map from node_idx to the index of its PTE in ptes list
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


        # Case 1: Both nodes are in the same PTE
        if pte1_idx == pte2_idx:
            pte, nodes = ptes[pte1_idx]
            
            new_pte = pte.self_trace(join_legs1, join_legs2)
            ptes[pte1_idx] = (new_pte, nodes)

            new_traceable_legs = [
                leg
                for leg in traceable_legs[node_idx1]
                if leg not in join_legs1 and leg not in join_legs2
            ]

            for node in nodes:
                traceable_legs[node] = new_traceable_legs

            open_legs_set = set(new_traceable_legs)
            open_leg_indices = [i for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_indices += [i + (new_pte.h.shape[1]//2) for i, leg in enumerate(new_pte.legs) if leg in open_legs_set]
            open_leg_submatrix = new_pte.h[:, open_leg_indices]
            rank_submatrix = tn.compute_gf2_rank(open_leg_submatrix)
            
        # Case 2: Nodes are in different PTEs - merge them
        else:
            pte1, nodes1 = ptes[pte1_idx]
            pte2, nodes2 = ptes[pte2_idx]

            new_pte = pte1.conjoin(pte2, legs1=join_legs1, legs2=join_legs2)
            merged_nodes = nodes1.union(nodes2)
            # Update the first PTE with merged result
            ptes[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            ptes.pop(pte2_idx)

            # Update node_to_pte mappings
            for node in nodes2:
                node_to_pte[node] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node] = pte_idx - 1

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
            rank_submatrix = tn.compute_gf2_rank(open_leg_submatrix)

            for node in merged_nodes:
                traceable_legs[node] = new_traceable_legs

        total_cost += 2**rank_submatrix

    return total_cost

def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


with open("custom_cost_calc_ds.csv", "w") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["distance", "name", "num_run", "custom_cost"])

ds = [3, 4, 5, 7, 9]
num_runs = 100

for distance in ds:
    coloring = generate_checkerboard_coloring(distance)
    compass_code = CompassCode(distance, coloring)
    
    for i in range(num_runs):
        for name, rep in compass_code.get_representations().items():
            tn = rep()
            custom_cost = run_upper_bound_calc(tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)
            
            with open(f"custom_cost_calc_ds.csv", "a") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([distance, name, i, custom_cost])
            