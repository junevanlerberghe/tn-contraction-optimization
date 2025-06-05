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
    tn.analyze_traces(cotengra=True, minimize="size", max_repeats=150)
    start = time.time()
    wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = stabilizer_enumerator_polynomial(
        tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
    )
    end = time.time()
    return end - start, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count


def reorder_traces_by_sparsity(tn, reverse=False):
    """
    Compute sparsity = (nnz of node1.h) + (nnz of node2.h) for each trace,
    then sort tn.traces ascending by that sum and assign back.
    """
    trace_scores = []
    for trace in tn.traces:
        node_idx1, node_idx2, join_legs1, join_legs2 = trace
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity = np.count_nonzero(h1) + np.count_nonzero(h2)
        trace_scores.append((sparsity, trace))

    # sort by the sparsity (the first element of each tuple)
    trace_scores.sort(key=lambda x: x[0], reverse=reverse)

    # extract just the traces in sorted order
    tn.traces = [t for (_, t) in trace_scores]


def run_wep_for_sorted_traces(coloring, d):
    compass_code = CompassCode(d, coloring)
    tn = compass_code.concatenated()

    # Run wep calculation for sorted pcm sparsity
    sparsity_info = []
    for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity_info.append(np.count_nonzero(h1) + np.count_nonzero(h2))

    numerator = sum((i+1) * w for i, w in enumerate(sparsity_info))
    denominator = len(sparsity_info)
    weighted_avg = round(numerator / denominator, 3)

    reorder_traces_by_sparsity(tn)
    contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes = get_contraction_time(tn)
    avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)


    with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([coloring, d, contraction_time, False, contraction_cost, contraction_width, weighted_avg, avg_intermediate_tensor_size, np.max(intermediate_tensor_sizes)])

    # Run wep calculation for reverse sorted pcm sparsity
    tn = compass_code.concatenated()

    sparsity_info = []
    for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity_info.append(np.count_nonzero(h1) + np.count_nonzero(h2))


    numerator = sum(i * w for i, w in enumerate(sparsity_info))
    numerator = sum((i+1) * w for i, w in enumerate(sparsity_info))
    denominator = len(sparsity_info)
    
    reorder_traces_by_sparsity(tn, reverse=True)
    contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes = get_contraction_time(tn)
    avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

    with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([coloring, d, contraction_time, True, contraction_cost, contraction_width, weighted_avg, avg_intermediate_tensor_size, np.max(intermediate_tensor_sizes)])

    # Run wep calculation for randomly sorted pcm sparsity
    tn = compass_code.concatenated()

    
    contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes = get_contraction_time(tn)
    avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

    with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([coloring, d, contraction_time, True, contraction_cost, contraction_width, weighted_avg, avg_intermediate_tensor_size, np.max(intermediate_tensor_sizes)])

    

def run_wep_for_many_traces(coloring, d):
    compass_code = CompassCode(d, coloring)

    #for name, rep in compass_code.get_representations().items():
    for _ in range(30):
        tn = compass_code.concatenated()
        np.random.shuffle(tn.traces)

        # need to loop through the traces and analyze cost somehow
        sparsity_info = []
        for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
            h1 = np.array(tn.nodes[node_idx1].h)
            h2 = np.array(tn.nodes[node_idx2].h)
            sparsity_info.append(np.count_nonzero(h1) + np.count_nonzero(h2))

        numerator = sum((i+1) * w for i, w in enumerate(sparsity_info))
        weighted_avg = round(numerator / len(sparsity_info), 3)

        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('sparsity_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([d, "concatenated", round(contraction_time, 3), contraction_cost, contraction_width, weighted_avg, avg_intermediate_tensor_size, total_ops_count])


with open('sparsity_tests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["distance", "representation", "contraction_time", "contraction_cost", "contraction_width", "sparsity_weighted_avg", "avg_intermediate_tensor_size", "total_operations"])


def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


#### Should think about how I want to run this in the future, wrap it all up into a nice function 
    ### with an option to name the output csv file as well
ds = [4]
q_shors = [1.0]

for d in ds:
    for q_shor in q_shors:
        coloring = generate_checkerboard_coloring(d)
        for i in range(d - 1):
            for j in range(d - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2
        run_wep_for_many_traces(coloring, d)


# for _ in range(10):
#     coloring = generate_checkerboard_coloring(3)
#     run_wep_for_sorted_traces(coloring, 3)