import argparse
import csv
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "planqtn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from planqtn.networks.holographic_happy_code import HolographicHappyTN
from planqtn.contraction_visitors.max_size_cost_visitor import MaxTensorSizeCostVisitor
from planqtn.contraction_visitors.sparsity_visitor import SparsityVisitor
from planqtn.contraction_visitors.upper_bound_cost_visitor import UpperBoundCostVisitor
from planqtn.linalg import gauss
from repetition_tree_code import RepCodeTreeConcatenatedTN
from cotengra import OptimalOptimizer
from planqtn.contraction_visitors.stabilizer_flops_cost_fn import (
    StabilizerCodeFlopsCostVisitor,
)
from planqtn.networks.rotated_surface_code import RotatedSurfaceCodeTN
from planqtn.tensor_network import Contraction


def get_intial_bruteforce_cost(tn):
    brute_force_operations = 0
    for node_idx, node in tn.nodes.items():
        h_reduced = gauss(node.h)
        h_reduced = h_reduced[~np.all(h_reduced == 0, axis=1)]
        r = len(h_reduced)
        brute_force_operations += 2**r
    return brute_force_operations


def get_optimal_cost(tn):
    initial_bruteforce = get_intial_bruteforce_cost(tn)

    contraction = Contraction(
        tn,
        lambda node: node.copy(),
    )
    contraction_copy = Contraction(
        tn,
        lambda node: node.copy(),
    )

    opt = OptimalOptimizer(minimize="custom_flops", contraction_info=contraction_copy)
    search_params = {
        "optimal_minimizer": "custom_flops",
        "contraction_info": contraction_copy,
    }

    start = time.time()
    contraction._cot_tree = opt.search(
        contraction.inputs, contraction.output, contraction.size_dict, search_params
    )
    end = time.time()

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    upper_bound_visitor = UpperBoundCostVisitor()
    sparsity_visitor = SparsityVisitor()
    max_size_visitor = MaxTensorSizeCostVisitor()

    contraction.contract(visitors=[cost_visitor, upper_bound_visitor, sparsity_visitor, max_size_visitor], cotengra=False)

    upper_bound_cost = upper_bound_visitor.total_cost + initial_bruteforce
    contraction_cost = cost_visitor.total_cost + initial_bruteforce

    return (
        upper_bound_cost,
        cost_visitor.total_cost,
        contraction_cost,
        max_size_visitor.max_size,
        sparsity_visitor.tensor_sparsity,
        end - start,
    )


def find_optimal_costs(file_name="optimal_costs.csv"):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "method",
                    "tensor_network",
                    "num_qubits",
                    "upper_bound_operations",
                    "operations",
                    "operations_with_bruteforce",
                    "max_tensor_size",
                    "avg_sparsity",
                    "duration",
                ]
            )

    # These codes were the only ones that were able to finish in a reasonable amount of time
    for layer in [2, 3]:
        print("Finding optimal cost for Concatenated Repetition code with", layer, "layers")
        tn = RepCodeTreeConcatenatedTN(layer)
        (
            upper_bount_cost,
            custom_cost,
            cost_with_bruteforce,
            max_size,
            tensor_sparsities,
            duration,
        ) = get_optimal_cost(tn)

        sparsities = [s[-1] for s in tensor_sparsities]
        avg_sparsity = np.round(np.mean(sparsities), 5)
        with open(file_name, "a") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "optimal",
                    "Concatenated Repetition",
                    3**layer,
                    upper_bount_cost,
                    custom_cost,
                    cost_with_bruteforce,
                    max_size,
                    avg_sparsity,
                    duration,
                ]
            )

    for distance in [3]:
        print("Finding optimal cost for Rotated Surface code with distance: ", distance)
        tn = RotatedSurfaceCodeTN(distance)
        (
            upper_bount_cost,
            custom_cost,
            cost_with_bruteforce,
            max_size,
            tensor_sparsities,
            duration,
        ) = get_optimal_cost(tn)

        sparsities = [s[-1] for s in tensor_sparsities]
        avg_sparsity = np.round(np.mean(sparsities), 5)
        with open(file_name, "a") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "optimal",
                    "Rotated Surface Code",
                    distance**2,
                    upper_bount_cost,
                    custom_cost,
                    cost_with_bruteforce,
                    max_size,
                    avg_sparsity,
                    duration,
                ]
            )

    for layer in [2]:
        print("Finding optimal cost for Holographic code with layers: ", layer)
        tn = HolographicHappyTN(layer)
        (
            upper_bount_cost,
            custom_cost,
            cost_with_bruteforce,
            max_size,
            tensor_sparsities,
            duration,
        ) = get_optimal_cost(tn)

        sparsities = [s[-1] for s in tensor_sparsities]
        avg_sparsity = np.round(np.mean(sparsities), 5)
        with open(file_name, "a") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "optimal",
                    "Holographic",
                    25,
                    upper_bount_cost,
                    custom_cost,
                    cost_with_bruteforce,
                    max_size,
                    avg_sparsity,
                    duration,
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimal contraction cost experiments."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="optimal_contraction_costs.csv",
        help="Name of the CSV file to save results.",
    )

    args = parser.parse_args()

    find_optimal_costs(file_name=args.file_name)
