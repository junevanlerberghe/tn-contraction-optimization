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

from planqtn.networks.holographic_happy_code import HolographicHappyTN
from planqtn.networks.stabilizer_tanner_code import StabilizerTannerCodeTN
from planqtn.contraction_visitors.max_size_cost_visitor import MaxTensorSizeCostVisitor

from planqtn.linalg import gauss
from repetition_tree_code import RepCodeTreeConcatenatedTN

from planqtn.contraction_visitors.sparsity_visitor import SparsityVisitor
from planqtn.networks.rotated_surface_code import RotatedSurfaceCodeTN
from planqtn.networks.stabilizer_measurement_state_prep import (
    StabilizerMeasurementStatePrepTN,
)

from planqtn.tensor_network import TensorNetwork
from planqtn.contraction_visitors.stabilizer_flops_cost_fn import (
    StabilizerCodeFlopsCostVisitor,
)
from planqtn.contraction_visitors.upper_bound_cost_visitor import UpperBoundCostVisitor

from bb_parity_check import create_full_parity_check, get_bb_params
from utils import generate_hamming_parity_check
from planqtn.progress_reporter import TqdmProgressReporter
from planqtn.progress_reporter import DummyProgressReporter, ProgressReporter


def find_contraction_cost(
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
    """Finds the contraction cost of a tensor network using Cotengra to find a contraction schedule.
    Uses ContractionVisitors to collect various information about the contraction, such as cost.

    Args:
        tn (TensorNetwork): The tensor network to find contraction schedule for.
        minimize (str): The cost function to minimize when finding the contraction schedule.
        methods (List[str]): List of methods to use for contraction (e.g., greedy, kahypar).
        search_params (Dict): Additional search parameters for cotengra.
        open_legs (List[Tuple[int, int]]): List of open legs to consider in the cost calculation. Each leg is a tuple of (node_index, leg_index).
        verbose (bool): Whether to print detailed information during the process.
        progress_reporter (ProgressReporter): An instance of ProgressReporter to report progress.
        cotengra (bool): Whether to use cotengra to find the contraction schedule.
        max_repeats (int): Maximum number of trials to allow Cotengra to run for each code.
        max_time (int): Maximum time (in seconds) to allow Cotengra to run for each code.

    Returns:
        Tuple[int, int, int, int, List[float], float]: A tuple containing the information from the ContractionVisitors
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

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    upper_bound_visitor = UpperBoundCostVisitor()
    sparsity_visitor = SparsityVisitor()
    max_size_visitor = MaxTensorSizeCostVisitor()

    start = time.time()
    tn.conjoin_nodes(
        verbose=verbose,
        visitors=[
            cost_visitor,
            upper_bound_visitor,
            sparsity_visitor,
            max_size_visitor,
        ],
        cotengra=cotengra,
        open_legs=open_legs,
        cotengra_opts={
            "minimize": minimize,
            "methods": methods,
            "max_time": max_time,
            "max_repeats": max_repeats,
        },
        progress_reporter=progress_reporter,
        search_params=search_params,
    )
    end = time.time()
    upper_bound_cost = upper_bound_visitor.total_cost + brute_force_operations
    contraction_cost = cost_visitor.total_cost + brute_force_operations

    return (
        upper_bound_cost,
        cost_visitor.total_cost,
        contraction_cost,
        max_size_visitor.max_size,
        sparsity_visitor.tensor_sparsity,
        end - start,
    )


def run_contraction_cost_experiment(
    networks: Dict[Tuple[str, int], TensorNetwork],
    num_runs,
    file_name,
    minimize="custom",
    methods=["greedy"],
    search_params={},
    max_repeats=128,
    max_time=None,
):
    """Run contraction cost experiments for the given tensor networks and save results to a CSV file.

    Args:
        networks (Dict[Tuple[str, int], TensorNetwork]): A dictionary mapping (name, num_qubits) to tensor networks.
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
        minimize (str): The cost function to minimize when finding the contraction schedule.
        methods (List[str]): List of methods to use for contraction (e.g., greedy, kahypar).
        search_params (Dict): Additional search parameters for cotengra.
        max_repeats (int): Maximum number of trials to allow Cotengra to run for each code.
        max_time (int): Maximum time (in seconds) to allow Cotengra to run for each code.
    """

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
                    "upper_bound_cost",
                    "operations",
                    "operations_with_bruteforce",
                    "max_tensor_size",
                    "avg_tensor_sparsity",
                    "time",
                ]
            )

    for i in range(num_runs):
        for key, creation_fn in networks.items():
            name, num_qubits = key
            tn = creation_fn()

            print(f"Finding contraction cost for {name}, run {i+1}")

            (
                upper_bound_cost,
                custom_cost,
                cost_with_bruteforce,
                max_tensor_size,
                tensor_sparsities,
                cotengra_duration,
            ) = find_contraction_cost(
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

            sparsities = [s[-1] for s in tensor_sparsities]

            with open(file_name, "a") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(
                    [
                        minimize,
                        methods,
                        name,
                        num_qubits,
                        i,
                        upper_bound_cost,
                        custom_cost,
                        cost_with_bruteforce,
                        max_tensor_size,
                        round(np.mean(sparsities), 5),
                        cotengra_duration,
                    ]
                )


def make_all_tensor_networks(
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
    ]
):
    """Makes a dictionary of all tensor networks to be used in the experiments.

    Args:
        codes: List of codes to create. Default is all codes.
    Returns:
        Dictionary mapping (name, num_qubits) to tensor network creation functions.
    """
    tensor_networks = {}

    for layer in [2, 3, 4]:
        # Symmetric Tree Tensor Network (Concatenated Repetition):
        if "concatenated" in codes:
            tensor_networks[("Concatenated Repetition", 3**layer)] = (
                lambda layer=layer: RepCodeTreeConcatenatedTN(layer)
            )

    for d in [3, 5, 7]:
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
    for r in [3, 4]:
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
        n_qubits = [25, 95, 355]
        for i in range(len(layers)):
            tensor_networks[("Holographic", n_qubits[i])] = lambda layer=layers[
                i
            ]: HolographicHappyTN(layer)

    # BB Code MSP
    bb_codes = [18, 30]

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


def run_all_contraction_cost_experiments(
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
):
    """Create tensor networks and run contraction cost experiments for flops and custom SST minimize functions.

    Args:
        num_runs: Number of runs for each tensor network.
        file_name: Name of the CSV file to save results.
        methods: List of methods to use for contraction (e.g., greedy, kahypar).
        codes: List of codes to include in the experiment. Default is all codes.
        max_repeats: Maximum number of trials to allow Cotengra to run for each code.
        max_time: Maximum time (in seconds) to allow Cotengra to run for each code.
    """

    for method in methods:
        search_params = {}
        if method == "kahypar":
            search_params = {
                "greedy_minimizer": "custom_flops",
                "optimal_minimizer": "custom_flops",
                "sub_optimize_minimizer": "custom_flops",
            }

        # Run SST cost function. If Kahypar, also use flops for the internal greedy and optimal minimizers.
        tensor_networks = make_all_tensor_networks(codes)
        run_contraction_cost_experiment(
            tensor_networks,
            num_runs,
            file_name,
            minimize="custom_flops",
            methods=[method],
            search_params=search_params,
            max_repeats=max_repeats,
            max_time=max_time,
        )

        # Run default flops (dense) cost function with no customizations.
        tensor_networks = make_all_tensor_networks(codes)
        run_contraction_cost_experiment(
            tensor_networks,
            num_runs,
            file_name,
            minimize="flops",
            methods=[method],
            max_repeats=max_repeats,
            max_time=max_time,
        )


def find_sparsity_information(
    num_runs=10,
    file_name="tensor_sparsity_info.csv",
    minimize="custom_flops",
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
):
    tensor_networks = make_all_tensor_networks(codes)

    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "cost_fn",
                    "network",
                    "num_qubits",
                    "num_run",
                    "num_open_legs",
                    "actual_tensor_size",
                    "dense_tensor_size",
                    "tensor_sparsity",
                ]
            )

    for i in range(num_runs):
        for key, creation_fn in tensor_networks.items():
            name, num_qubits = key
            tn = creation_fn()

            print(f"Finding contraction cost for {name}, run {i+1}")
            _, _, _, _, tensor_sparsities, _ = find_contraction_cost(
                tn,
                minimize=minimize,
                verbose=False,
                progress_reporter=TqdmProgressReporter(),
                cotengra=True,
            )

            for open_legs, new_size, dense_size, sparsity in tensor_sparsities:
                with open(file_name, "a") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(
                        [
                            minimize,
                            name,
                            num_qubits,
                            i,
                            open_legs,
                            new_size,
                            dense_size,
                            sparsity,
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
        "--sparsity_collection",
        action="store_true",
        help="Collect tensor sparsity data instead of contraction cost",
    )
    args = parser.parse_args()

    if args.sparsity_collection:
        find_sparsity_information(
            num_runs=args.num_runs, file_name=args.file_name, codes=args.codes
        )
    else:
        run_all_contraction_cost_experiments(
            num_runs=args.num_runs,
            file_name=args.file_name,
            methods=args.methods,
            codes=args.codes,
            max_repeats=args.max_repeats,
            max_time=args.max_time,
        )
