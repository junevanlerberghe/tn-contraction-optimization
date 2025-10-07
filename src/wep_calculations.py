import argparse
import copy
import csv
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "planqtn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bb_parity_check import create_full_parity_check, get_bb_params
from planqtn.networks.holographic_happy_code import HolographicHappyTN
from planqtn.networks.stabilizer_tanner_code import StabilizerTannerCodeTN

from planqtn.linalg import gauss
from repetition_tree_code import RepCodeTreeConcatenatedTN

from planqtn.networks.rotated_surface_code import RotatedSurfaceCodeTN
from planqtn.networks.stabilizer_measurement_state_prep import (
    StabilizerMeasurementStatePrepTN,
)

from planqtn.tensor_network import (
    _PartiallyTracedEnumerator,
    Contraction,
    TensorNetwork,
)
from planqtn.contraction_visitors.stabilizer_flops_cost_fn import (
    StabilizerCodeFlopsCostVisitor,
)

from utils import generate_hamming_parity_check
from planqtn.progress_reporter import TqdmProgressReporter
from planqtn.progress_reporter import DummyProgressReporter, ProgressReporter
from planqtn.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator
from planqtn.operation_tracker import get_tracker


def find_wep(
    tn: TensorNetwork,
    minimize: str = "custom_flops",
    methods: List[str] = ["greedy"],
    search_params: Dict = {},
    open_legs: List[Tuple[int, int]] = [],
    verbose: bool = False,
    progress_reporter: ProgressReporter = DummyProgressReporter(),
    cotengra: bool = True,
    max_repeats: int = 128,
    max_time: int = None,
):
    """Finds the contraction cost of a tensor network using cotengra to find a contraction schedule.
    Calculates both the upper bound cost from the open legs and the number of operations it will take to
    contract the network.

    Args:
        tn (TensorNetwork): The tensor network to analyze.
        minimize (str): The cost function to minimize when finding the contraction schedule. Options are "flops", "size", "custom".
        open_legs (List[Tuple[int, int]]): List of open legs to consider in the cost calculation. Each leg is a tuple of (node_index, leg_index).
        verbose (bool): Whether to print detailed information during the process.
        progress_reporter (ProgressReporter): An instance of ProgressReporter to report progress.
        cotengra (bool): Whether to use cotengra to find the contraction schedule. If False, uses a default greedy algorithm.

    Returns:
        Tuple[int, int]: A tuple containing the upper bound cost and the contraction cost.
    """
    assert (
        progress_reporter is not None
    ), "Progress reporter must be provided, it is None"

    brute_force_operations = 0
    for node_idx, node in tn.nodes.items():
        h_reduced = gauss(node.h)
        h_reduced = h_reduced[~np.all(h_reduced == 0, axis=1)]
        r = len(h_reduced)
        brute_force_operations += 2**r

    print(
        "Running stabilizer enumerator polynomial with search params: ", search_params
    )
    get_tracker().reset()

    contraction = Contraction[_PartiallyTracedEnumerator](
        tn,
        lambda node: _PartiallyTracedEnumerator.from_stabilizer_code_tensor_enumerator(
            node, tn.truncate_length, verbose, progress_reporter, open_legs
        ),
    )

    cotengra_opts = {
        "minimize": minimize,
        "methods": methods,
        "max_time": max_time,
        "max_repeats": max_repeats,
    }
    start = time.time()
    final_tensor = contraction.contract(
        cotengra=cotengra,
        progress_reporter=progress_reporter,
        open_legs=open_legs,
        verbose=verbose,
        cotengra_opts=cotengra_opts,
        search_params=search_params,
    )
    end = time.time()
    real_operations = get_tracker().get()

    if len(final_tensor.tensor) > 1:
        wep = final_tensor.ordered_key_tensor(
            open_legs,
            progress_reporter=progress_reporter,
            verbose=verbose,
        )
    else:
        wep = final_tensor.tensor[()]
        wep = wep.normalize(verbose=verbose)

    found_tree = contraction._cot_tree
    print("tree used in wep contraction: ", found_tree)

    contraction = Contraction[StabilizerCodeTensorEnumerator](
        tn, lambda node: node.copy(), cotengra_tree=found_tree
    )
    cotengra_score = found_tree.get_score()

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    contraction.contract(
        cotengra=False,
        progress_reporter=progress_reporter,
        open_legs=open_legs,
        verbose=verbose,
        visitors=[cost_visitor],
    )
    custom_cost = cost_visitor.total_cost

    return (
        end - start,
        real_operations,
        round(2**cotengra_score),
        custom_cost,
        wep,
    )

def make_all_tensor_networks(codes):
    tensor_networks = {}

    for layer in [2, 3, 4, 5]:
        # Symmetric Tree Tensor Network (Concatenated Repetition):
        if "concatenated" in codes:
            tensor_networks[("Concatenated Repetition", 3**layer)] = (
                lambda layer=layer: RepCodeTreeConcatenatedTN(layer)
            )

    for d in [3, 5, 7, 9]:
        # Rotated Surface Code - [[d^2,1,d]]
        if "rotated" in codes:
            tensor_networks[("Rotated Surface", d**2)] = (
                lambda d=d: RotatedSurfaceCodeTN(d)
            )

    # MSP for Rotated Surface Code -- [[d^2,1,d]]
    for d in [3, 5]:
        H_surface = RotatedSurfaceCodeTN(d).conjoin_nodes().h

        if "rotated_msp" in codes:
            tensor_networks[("Rotated Surface MSP", d**2)] = (
                lambda H_surface=H_surface: StabilizerMeasurementStatePrepTN(H_surface)
            )

        if "rotated_tanner" in codes:
            tensor_networks[("Rotated Surface Tanner", d**2)] = (
                lambda H_surface=H_surface: StabilizerTannerCodeTN(H_surface)
            )

    # MSP for Hamming Code (non-degenerate) -- [[7,1,3]], [[15,7,3]], [[31,21,3]]
    for r in [3]:
        if "hamming_msp" in codes:
            tensor_networks[("Hamming MSP", 2**r - 1)] = (
                lambda r=r: StabilizerMeasurementStatePrepTN(
                    generate_hamming_parity_check(r)
                )
            )

        if "hamming_tanner" in codes:
            tensor_networks[("Hamming Tanner", 2**r - 1)] = (
                lambda r=r: StabilizerTannerCodeTN(generate_hamming_parity_check(r))
            )

    # Holographic Happy TN - [[25,11,3]], [[95,51,3]]
    if "holo" in codes:
        layers = [2, 3]
        n_qubits = [25, 95]
        for i in range(len(layers)):
            tensor_networks[("Holographic", n_qubits[i])] = lambda layer=layers[
                i
            ]: HolographicHappyTN(layer)

    #  # BB Code MSP
    bb_codes = [18]

    for i in range(len(bb_codes)):
        l, m, a, b = get_bb_params(bb_codes[i])
        H_bb = create_full_parity_check(l, m, a, b)

        if "bb_msp" in codes:
            tensor_networks[("BB MSP", bb_codes[i])] = (
                lambda H_bb=H_bb: StabilizerMeasurementStatePrepTN(H_bb)
            )

        if "bb_tanner" in codes:
            tensor_networks[("BB Tanner", bb_codes[i])] = (
                lambda H_bb=H_bb: StabilizerTannerCodeTN(H_bb)
            )

    return tensor_networks


def find_weps(
    num_runs=100,
    file_name="contraction_costs.csv",
    methods=["greedy", "kahypar"],
    codes=[
        "concatenated",
        "rotated",
        "rotated_msp",
        "rotated_tanner",
        "hamming_msp",
        "hamming_tanner",
        "holo",
        "bb_msp",
        "bb_tanner",
    ],
    max_repeats=128,
    max_time=None,
    minimize="flops",
):
    """Create tensor networks and run contraction cost experiments for combo and custom minimize functions.

    Args:
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
        tn (bool): Whether to include the given tensor network in the experiments.
    """

    for method in methods:
        search_params = {}
        if method == "kahypar":
            search_params = {
                "greedy_minimizer": "custom_flops",
                "optimal_minimizer": "custom_flops",
                "sub_optimize_minimizer": "custom_flops",
            }
        tensor_networks = make_all_tensor_networks(codes)

        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(
                    [
                        "cost_fn",
                        "methods",
                        "tensor_network",
                        "num_qubits",
                        "num_run",
                        "real_operations",
                        "cotengra_cost",
                        "custom_cost",
                        "time",
                        "wep",
                    ]
                )

        for i in range(num_runs):
            for key, creation_fn in tensor_networks.items():
                name, num_qubits = key
                tn = creation_fn()

                print(f"Finding contraction cost for {name}, run {i+1}")

                (duration, real_operations, cotengra_cost, custom_cost, wep) = find_wep(
                    tn,
                    minimize=minimize,
                    methods=methods,
                    verbose=False,
                    progress_reporter=TqdmProgressReporter(),
                    cotengra=True,
                    search_params=copy.deepcopy(search_params),
                    max_repeats=max_repeats,
                    max_time=max_time,
                )

                with open(file_name, "a") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(
                        [
                            minimize,
                            methods,
                            name,
                            num_qubits,
                            i,
                            real_operations,
                            cotengra_cost,
                            custom_cost,
                            duration,
                            wep,
                        ]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run contraction cost experiments.")
    parser.add_argument(
        "--file_name",
        type=str,
        default="contraction_costs.csv",
        help="Name of the CSV file to save results.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs for each tensor network.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["greedy", "kahypar"],
        help="List of methods to use for contraction (e.g., greedy, kahypar).",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=[
            "concatenated",
            "rotated",
            "rotated_msp",
            "rotated_tanner",
            "hamming_msp",
            "hamming_tanner",
            "holo",
            "bb_msp",
            "bb_tanner",
        ],
        help="List of methods to run (concatenated, rotated_surface, hamming, holo, bb)",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=None,
        help="Maximum time (in seconds) to allow Cotengra to run for each code.",
    )
    parser.add_argument(
        "--max_repeats",
        type=int,
        default=128,
        help="Maximum number of trials to allow Cotengra to run for each code.",
    )
    parser.add_argument(
        "--minimize",
        type=str,
        default="custom_flops",
        help="Minimize strategy to use (custom_flops, custom_size, flops, size, combo)",
    )

    args = parser.parse_args()

    find_weps(
        num_runs=args.num_runs,
        file_name=args.file_name,
        methods=args.methods,
        codes=args.codes,
        max_repeats=args.max_repeats,
        max_time=args.max_time,
        minimize=args.minimize,
    )
