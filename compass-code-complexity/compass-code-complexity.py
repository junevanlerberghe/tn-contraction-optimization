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
from compassCodes.compass_code_rotated import CompassCodeRotatedTN
from planqtn.networks.compass_code import CompassCodeConcatenateAndSparsifyTN, CompassCodeDualSurfaceCodeLayoutTN
from planqtn.networks.stabilizer_tanner_code import StabilizerTannerCodeTN
from planqtn.contraction_visitors.max_size_cost_visitor import MaxTensorSizeCostVisitor

from planqtn.linalg import gauss

from planqtn.contraction_visitors.sparsity_visitor import SparsityVisitor
from planqtn.networks.stabilizer_measurement_state_prep import (
    StabilizerMeasurementStatePrepTN,
)

from planqtn.tensor_network import TensorNetwork
from planqtn.contraction_visitors.stabilizer_flops_cost_fn import (
    StabilizerCodeFlopsCostVisitor,
)
from planqtn.contraction_visitors.upper_bound_cost_visitor import UpperBoundCostVisitor
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
    max_time: int = None
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

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    upper_bound_visitor = UpperBoundCostVisitor()
    sparsity_visitor = SparsityVisitor()
    max_size_visitor = MaxTensorSizeCostVisitor()

    print("Running conjoin nodes with search params: ", search_params)
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
        cotengra_opts = {
            "minimize": minimize, 
            "methods": methods, 
            "max_time": max_time, 
            "max_repeats": max_repeats
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
        end - start
    )


def run_contraction_cost_experiment(
    networks: Dict[Tuple[str, int], TensorNetwork],
    num_runs,
    file_name,
    minimize="custom",
    methods=["greedy"],
    search_params={},
    max_repeats=128,
    max_time=None
):
    """Run contraction cost experiments for the given tensor networks and save results to a CSV file.

    Args:
        networks (Dict[Tuple[str, int], TensorNetwork]): A dictionary mapping (name, num_qubits) to tensor networks.
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
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
                    "p_flip",
                    "num_run",
                    "upper_bound_cost",
                    "operations",
                    "operations_with_bruteforce",
                    "max_tensor_size",
                    "avg_tensor_sparsity",
                    "time"
                ]
            )

    for i in range(num_runs):
        for key, creation_fn in networks.items():
            name, num_qubits, p_flip = key
            tn = creation_fn()

            print(f"Finding contraction cost for {name}, run {i+1}")

            (
                upper_bound_cost,
                custom_cost,
                cost_with_bruteforce,
                max_tensor_size,
                tensor_sparsities,
                cotengra_duration
            ) = find_contraction_cost(
                tn,
                minimize=minimize,
                methods=methods,
                verbose=False,
                progress_reporter=TqdmProgressReporter(),
                cotengra=True,
                search_params=copy.deepcopy(search_params),
                max_repeats=max_repeats,
                max_time=max_time
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
                        p_flip,
                        i,
                        upper_bound_cost,
                        custom_cost,
                        cost_with_bruteforce,
                        max_tensor_size,
                        round(np.mean(sparsities), 5),
                        cotengra_duration
                    ]
                )

def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]


def make_all_tensor_networks(representations=["concat_sparsify", "qubit_wise", "dual_surface", "msp", "tanner"]):
    tensor_networks = {}

    for d in [3, 5, 7, 9, 11]:
        for p_flip in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # if random number is less than q_shor, then put an X stabilizer in that plaquette
            # start with distance d checkerboard coloring, then change to Xs whenver necessary
            coloring = generate_checkerboard_coloring(d)
            for i in range(d - 1):
                for j in range(d - 1):
                    if np.random.rand() < p_flip:
                        coloring[i][j] = 2
            
            if "concat_sparsify" in representations:
                tensor_networks[("Concat & Sparsify", d, p_flip)] = (
                    lambda coloring=coloring: CompassCodeConcatenateAndSparsifyTN(coloring)
                )
            
            if("qubit_wise" in representations):
                tensor_networks[("Qubit-wise", d, p_flip)] = (
                    lambda coloring=coloring: CompassCodeRotatedTN(coloring)
                )

            if("dual_surface" in representations):
                tensor_networks[("Dual Surface", d, p_flip)] = (
                    lambda coloring=coloring: CompassCodeDualSurfaceCodeLayoutTN(coloring)
                )

    for d in [3, 5, 7]:
        for p_flip in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # if random number is less than q_shor, then put an X stabilizer in that plaquette
            # start with distance d checkerboard coloring, then change to Xs whenver necessary
            coloring = generate_checkerboard_coloring(d)
            for i in range(d - 1):
                for j in range(d - 1):
                    if np.random.rand() < p_flip:
                        coloring[i][j] = 2

            tn_compass, _, _ = CompassCodeDualSurfaceCodeLayoutTN(coloring).conjoin_nodes()
            H_compass = tn_compass.h 

            if("msp" in representations):
                tensor_networks[("MSP", d, p_flip)] = (
                    lambda H_compass=H_compass: StabilizerMeasurementStatePrepTN(H_compass) 
                )

            if "tanner" in representations:
                tensor_networks[("Tanner", d, p_flip)] = (
                    lambda H_compass=H_compass: StabilizerTannerCodeTN(H_compass)
                )

    return tensor_networks

def run_compass_code_experiments(
    num_runs=100,
    file_name="contraction_costs.csv",
    representations=["concat_sparsify", "qubit_wise", "dual_surface", "msp", "tanner"],
):
    """Create tensor networks and run contraction cost experiments for combo and custom minimize functions.

    Args:
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
        tn (bool): Whether to include the given tensor network in the experiments.
    """

    tensor_networks = make_all_tensor_networks(representations)
    run_contraction_cost_experiment(
        tensor_networks, num_runs, file_name, minimize="custom_flops", methods=["greedy"], max_repeats=128
    )


def find_sparsity_information(
    num_runs=10, file_name="tensor_sparsity_info.csv", minimize="custom_flops", representations=["concat_sparsify", "qubit_wise", "dual_surface", "msp", "tanner"]
):
    tensor_networks = make_all_tensor_networks(representations)

    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["cost_fn", "network", "num_qubits", "num_run", "num_open_legs", "actual_tensor_size", "dense_tensor_size", "tensor_sparsity"]
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
                    writer.writerow([minimize, name, num_qubits, i, open_legs, new_size, dense_size, sparsity])


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
        "--representations", 
        nargs="+", 
        default=["concat_sparsify", "qubit_wise", "dual_surface", "msp", "tanner"],
        help="List of representations to run"
    )
    parser.add_argument(
        "--sparsity_collection",
        action='store_true',
        help="Collect tensor sparsity data instead of contraction cost"
    )
    args = parser.parse_args()
    
    if(args.sparsity_collection):
        find_sparsity_information(
            num_runs=args.num_runs,
            file_name=args.file_name,
            representations=args.representations
        )
    else:
        run_compass_code_experiments(
            num_runs=args.num_runs,
            file_name=args.file_name,
            representations=args.representations,
        )
