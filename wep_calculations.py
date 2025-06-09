import csv

from collections import defaultdict
import time
from typing import List, Tuple
import numpy as np
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.stabilizer_tensor_enumerator import (
    _index_leg,
)

from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator

#with open('intermediate_info.csv', 'w', newline='') as csvfile:
#    headers = ["q_shor", "coloring", "representation", "run #", "distance", "time", "contraction cost", "contraction width"]
#    writer = csv.writer(csvfile, delimiter=';')
#    writer.writerow(headers)

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
                pte = pte.tensor_product(pte2, verbose=verbose)

        if len(pte.tensor) > 1:
            tn._wep = pte.ordered_key_tensor(open_legs)
        else:
            tn._wep = pte.tensor[()]
            tn._wep = tn._wep.normalize(verbose=verbose)

        return tn._wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count


def get_contraction_time(tn):
    tn.analyze_traces(cotengra=True, minimize="combo", max_repeats=150)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = stabilizer_enumerator_polynomial(
        tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count


def run_wep_calc(q_shor, num_runs, coloring, distance, file_name):
    data = []
    compass_code = CompassCode(distance, coloring)
    for name, rep in compass_code.get_representations().items():
        # MSP and Tanner Network take too long for distances > 4, so skip them
        if(name == "Measurement State Prep" or name == "Tanner Network"):
            if(distance > 4):
                continue

        if(name == "Concatenated"):
            if(distance > 5):
                continue

        total_contraction_width = 0
        total_contraction_cost = 0
        total_ops = 0
        last_wep = None
        durations = []

        for i in range(num_runs):
            tn = rep()
            duration, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, operations = get_contraction_time(tn)
            durations.append(duration)
            total_contraction_width += contraction_width
            total_contraction_cost += contraction_cost
            total_ops += operations
            last_wep = wep 

            #with open('intermediate_info.csv', 'a', newline='') as csvfile:
            #    writer = csv.writer(csvfile, delimiter=';')
            #    writer.writerow([q_shor, coloring, name, i, distance, duration, contraction_cost, contraction_width])

        durations = np.array(durations)
        average_time = round(np.mean(durations), 3)
        median_time = round(np.median(durations), 3)
        std_dev = round(np.std(durations, ddof = 1), 3)

        average_contraction_width = round(total_contraction_width / num_runs, 3)
        average_contraction_cost = round(total_contraction_cost / num_runs, 3)
        average_operations = round(total_ops / num_runs, 3)
        average_tensor_size = round(sum(intermediate_tensor_sizes) / len(intermediate_tensor_sizes), 3)
        max_tensor = max(intermediate_tensor_sizes)

        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([q_shor, coloring, name, distance, average_time, median_time, std_dev, average_contraction_cost, average_contraction_width, average_operations, average_tensor_size, max_tensor, wep])
    
    return average_time, last_wep


def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


#### Should think about how I want to run this in the future, wrap it all up into a nice function 
    ### with an option to name the output csv file as well
file_name = 'wep_calcs.csv'

headers = ["q_shor", "coloring", "representation", "distance", "avg time", "median time", "std dev", "contraction cost", "contraction width", "operations", "avg intermediate tensor", "max tensor size", "wep"]
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(headers)

num_runs = 1
ds = [3, 4, 5, 6]
q_shors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for d in ds:
    for q_shor in q_shors:
        # if random number is less than q_shor, then put an X stabilizer in that plaquette
        # start with distance d checkerboard coloring, then change to Xs whenver necessary
        coloring = generate_checkerboard_coloring(d)
        for i in range(d - 1):
            for j in range(d - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2
        run_wep_calc(q_shor, num_runs, coloring, d, file_name)
