import ast
import csv
import os

import qlego
import sys
sys.path.insert(0, "/Users/junevanlerberghe/Documents/Duke/research/repositories/compass-wep-complexity/tnqec")


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

import qlego
print(qlego.__file__)

from compassCodes.compass_code import CompassCode
from qlego.tensor_network import _PartiallyTracedEnumerator

print(qlego.tensor_network.__file__)

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
                pte, ops_count, _ = node1_pte.self_trace(
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
                pte, ops_count, _ = node1_pte.merge_with(
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


'''def run_wep_calc(q_shor, num_runs, coloring, distance, file_name, intermediate_file="intermediate_info.csv"):
    compass_code = CompassCode(distance, coloring)

    completed_runs = defaultdict(list)
    coloring_map = {}

    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                key = (row[0], row[2], row[4])  # (q_shor, name, distance)
                completed_runs[key].append(row)
                coloring_map[(row[0], row[4])] = row[1]

    for name, rep in compass_code.get_representations().items():
        key = (str(q_shor), name, str(distance))
        existing_rows = completed_runs.get(key, [])
        runs_done = len(existing_rows)

        durations = [float(row[5]) for row in existing_rows]
        contraction_costs = [float(row[6]) for row in existing_rows]
        contraction_widths = [float(row[7]) for row in existing_rows]
        operations = [float(row[8]) for row in existing_rows]
        tensor_sizes = [float(row[9]) for row in existing_rows]

        if runs_done >= num_runs:
            print(f"Skipping {key}: {runs_done} runs already completed.")
            wep = existing_rows[-1][-1]

        else:
            print(f"Continuing {key}: {runs_done}/{num_runs} runs completed.")

            if((str(q_shor), str(distance)) in coloring_map):
                coloring = ast.literal_eval(coloring_map[(str(q_shor), str(distance))])
                compass_code = CompassCode(distance, list(coloring))
                rep = compass_code.get_representations()[name]

            print(f"\t with coloring: ", coloring)
            
            wep = None
            for i in range(runs_done, num_runs):

                coloring = generate_checkerboard_coloring(d)
                for i in range(d - 1):
                    for j in range(d - 1):
                        if np.random.rand() < q_shor:
                            coloring[i][j] = 2
                tn = rep()
                duration, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, ops = get_contraction_time(tn)
                
                durations.append(duration)
                contraction_widths.append(contraction_width)
                contraction_costs.append(contraction_cost)
                operations.append(ops)
                tensor_sizes.append(np.mean(intermediate_tensor_sizes))  # average per run

                with open(intermediate_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow([q_shor, coloring, name, i, distance, duration, contraction_cost, contraction_width, ops, np.mean(intermediate_tensor_sizes), wep])

            # Compute stats once num_runs are achieved
            if len(durations) == num_runs:
                average_time = round(np.mean(durations), 3)
                median_time = round(np.median(durations), 3)
                std_dev = round(np.std(durations, ddof=1), 3)

                average_contraction_cost = round(np.mean(contraction_costs), 3)
                average_contraction_width = round(np.mean(contraction_widths), 3)
                average_operations = round(np.mean(operations), 3)
                average_tensor_size = round(np.mean(tensor_sizes), 3)
                max_tensor = round(max(tensor_sizes), 3)

                with open(file_name, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow([q_shor, coloring, name, distance, average_time, median_time, std_dev, average_contraction_cost, average_contraction_width, average_operations, average_tensor_size, max_tensor, wep])

    return average_time, wep'''

def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]

def run_wep_calc(q_shor, num_runs, distance, file_name, intermediate_file="intermediate_info.csv"):
    completed_runs = defaultdict(list)

    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                key = (row[0], row[2], row[4], row[3])  # (q_shor, name, distance, run_index)
                completed_runs[key] = row

    for run_index in range(num_runs):
        # Generate a new coloring per run
        coloring = generate_checkerboard_coloring(distance)
        for i in range(distance - 1):
            for j in range(distance - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2

        compass_code = CompassCode(distance, coloring)

        for name, rep in compass_code.get_representations().items():
            key = (str(q_shor), name, str(distance), str(run_index))
            if key in completed_runs:
                print(f"Skipping run {run_index} for {name} (already completed)")
                continue

            print(f"Running {name}, run {run_index}, coloring: {coloring}")

            tn = rep()
            duration, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, ops = get_contraction_time(tn)

            with open(intermediate_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([
                    q_shor, coloring, name, run_index, distance, duration,
                    contraction_cost, contraction_width, ops,
                    np.mean(intermediate_tensor_sizes), wep
                ])

    # After all runs, aggregate the data per representation
    all_data = defaultdict(list)
    with open(intermediate_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            key = (row[0], row[2], row[4])  # (q_shor, name, distance)
            if row[0] == str(q_shor) and row[4] == str(distance):
                all_data[key].append(row)

    existing_summary_keys = set()
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                # Make sure the file is not empty or malformed
                if len(row) >= 4:
                    key = (row[0], row[1], row[2])  # (q_shor, name, distance)
                    existing_summary_keys.add(key)

    for key, rows in all_data.items():
        if len(rows) < num_runs:
            print(f"Not enough runs for {key}, skipping stats")
            continue

        if key in existing_summary_keys:
            print(f"Skipping {key}: already summarized in {file_name}")
            continue

        durations = [float(row[5]) for row in rows]
        contraction_costs = [float(row[6]) for row in rows]
        contraction_widths = [float(row[7]) for row in rows]
        operations = [float(row[8]) for row in rows]
        tensor_sizes = [float(row[9]) for row in rows]
        wep = rows[-1][-1]  # last WEP value

        average_time = round(np.mean(durations), 3)
        median_time = round(np.median(durations), 3)
        std_dev = round(np.std(durations, ddof=1), 3)

        average_contraction_cost = round(np.mean(contraction_costs), 3)
        average_contraction_width = round(np.mean(contraction_widths), 3)
        average_operations = round(np.mean(operations), 3)
        average_tensor_size = round(np.mean(tensor_sizes), 3)
        max_tensor = round(max(tensor_sizes), 3)

        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                key[0], key[1], key[2],
                average_time, median_time, std_dev,
                average_contraction_cost, average_contraction_width,
                average_operations, average_tensor_size,
                max_tensor, wep
            ])





#### Should think about how I want to run this in the future, wrap it all up into a nice function 
    ### with an option to name the output csv file as well
file_name = 'wep_calcs_mem_test.csv'

headers = ["q_shor", "representation", "distance", "avg time", "median time", "std dev", "contraction cost", "contraction width", "operations", "avg intermediate tensor", "max tensor size", "wep"]
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(headers)

num_runs = 10
ds = [3]
q_shors = [0.0]

for d in ds:
    for q_shor in q_shors:
        # if random number is less than q_shor, then put an X stabilizer in that plaquette
        # start with distance d checkerboard coloring, then change to Xs whenver necessary
        run_wep_calc(q_shor, num_runs, d, file_name, "mem_test.csv")
