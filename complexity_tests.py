from collections import defaultdict
import time
from typing import List, Tuple
from galois import GF2
import numpy as np
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.parity_check import conjoin, self_trace, sprint, sstr, tensor_product

from qlego.stabilizer_tensor_enumerator import (
    _index_leg,
)

import logging

from compassCodes.compass_code import CompassCode
from qlego.codes.rotated_surface_code import RotatedSurfaceCodeTN
from qlego.tensor_network import _PartiallyTracedEnumerator

# Configure logging
logging.basicConfig(
    filename="logs/compass_code_results.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting WEP calculations...")

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
            # tree.plot_contractions()
        summed_legs = [leg for leg in free_legs if leg not in open_legs]

        if len(tn.traces) == 0 and len(tn.nodes) == 1:
            return list(tn.nodes.items())[0][1].stabilizer_enumerator_polynomial(
                verbose=verbose,
                progress_reporter=progress_reporter,
                truncate_length=tn.truncate_length,
            )

        for node_idx1, node_idx2, join_legs1, join_legs2 in progress_reporter.iterate(
            traces, f"Tracing {len(traces)} legs", len(traces)
        ):
            node1_pte = None if node_idx1 not in tn.ptes else tn.ptes[node_idx1]
            node2_pte = None if node_idx2 not in tn.ptes else tn.ptes[node_idx2]

            if node1_pte == node2_pte:
                # both nodes are in the same PTE!
                pte = node1_pte.self_trace(
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
            else:
                pte = node1_pte.merge_with(
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
                pte = pte.tensor_product(pte2, verbose=verbose)

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes


def get_contraction_time(tn):
    tn.analyze_traces(cotengra=True, minimize="combo", max_repeats=150)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes = stabilizer_enumerator_polynomial(
        tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes


def run_wep_calc(num_runs, tn_creation_function):
    total_time = 0.0
    total_contraction_width = 0
    total_contraction_cost = 0
    last_wep = None
    for _ in range(num_runs):
        tn = tn_creation_function()
        duration, wep, contraction_width, contraction_cost, intermediate_tensor_sizes = get_contraction_time(tn)
        total_time += duration
        total_contraction_width += contraction_width
        total_contraction_cost += contraction_cost
        last_wep = wep 

    average_time = total_time / num_runs
    average_contraction_width = total_contraction_width / num_runs
    average_contraction_cost = total_contraction_cost / num_runs
    average_tensor_size = sum(intermediate_tensor_sizes) / len(intermediate_tensor_sizes)
    max_tensor = max(intermediate_tensor_sizes)

    logging.info(f"\tAverages over {num_runs} runs:")
    logging.info(f"\t\t Time: {average_time:.5f} s")
    logging.info(f"\t\t Contraction Width: {average_contraction_width}")
    logging.info(f"\t\t Contraction Cost: {average_contraction_cost}")
    logging.info(f"\t\t Average Intermediate Tensor Size: {average_tensor_size:.2f}")
    logging.info(f"\t\t Max Intermediate Tensor Size: {max_tensor}")
    logging.info(f"\tLast computed WEP: {wep}")
    return average_time, last_wep



num_runs = 1
colorings = []

colorings.append([[2, 2], [2, 2]])  #Shor's
colorings.append([[2, 1], [1, 2]]) # Rotated Surface Code
colorings.append([[2, 1], [2, 2]])
colorings.append([[1, 1], [1, 1]])

for coloring in colorings:
    compass_code = CompassCode(3, coloring)
    logging.info("------------------------------------")
    logging.info(f"Running WEP Calculation for Compass Code with coloring: {coloring}")
    for name, rep in compass_code.get_representations().items():
        logging.info("\t------------")
        logging.info(f"\t{name}")
        run_wep_calc(num_runs, rep)


#coloring = [[2, 2, 2, 2, 2], [2, 1, 2, 2, 2], [1, 2, 1, 1, 2], [2, 2, 2, 2, 1], [2, 2, 2, 2, 2]]
#coloring = [[2, 2], [2, 1]]
#compass = CompassCode(3, coloring)
#tn = compass.dual_surface()
#wep = tn.stabilizer_enumerator_polynomial(
#        verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
#    )
#print(wep)

'''
tn = RotatedSurfaceCodeTN(3)
wep = tn.stabilizer_enumerator_polynomial(
        verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
    )
print(wep)
'''

'''
def generate_checkerboard_coloring(d):
    return [[2 - (i + j) % 2 for j in range(d-1)] for i in range(d-1)]

d = 3
q_shors = [0.0, 0.3, 0.5, 0.8, 1.0]

for q_shor in q_shors:
    # if random number is less than q_shor, then put an X stabilizer in that plaquette
    # start with distance d checkerboard coloring, then change to Xs whenver necessary
    coloring = generate_checkerboard_coloring(d)
    for i in range(d - 1):
        for j in range(d - 1):
            if np.random.rand() < q_shor:
                coloring[i][j] = 2

    print("coloring for compass code with q_shor = ", q_shor, " : ", coloring)
    #code = CompassCode(d, coloring)
'''

