import csv
import sys
import os
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'planqtn'))
from planqtn.linalg import gauss

from compassCodes.compass_code_tnqec import CompassCodeRotatedRectangularTN
from planqtn.networks.compass_code import CompassCodeConcatenateAndSparsifyTN
from planqtn.networks.holographic_happy import HolographicHappyTN
from planqtn.networks.rotated_surface_code import RotatedSurfaceCodeTN
from planqtn.networks.stabilizer_measurement_state_prep import StabilizerMeasurementStatePrepTN

from planqtn.tensor_network import TensorNetwork
from planqtn.stabilizer_code_cost_fn import StabilizerCodeCostVisitor
from planqtn.upper_bound_cost_visitor import UpperBoundCostVisitor

from utils import generate_hamming_parity_check, generate_checkerboard_coloring
from bb_parity_check import create_coprime_parity_check, create_full_parity_check

from planqtn.progress_reporter import TqdmProgressReporter
from planqtn.progress_reporter import DummyProgressReporter, ProgressReporter
from planqtn.stabilizer_tensor_enumerator import _index_leg

def find_contraction_cost(
        tn: TensorNetwork,
        minimize: str = "custom",
        open_legs: List[Tuple[int, int]] = [],
        verbose: bool = False,
        progress_reporter: ProgressReporter = DummyProgressReporter(),
        cotengra: bool = True,
    ):
        """ Finds the contraction cost of a tensor network using cotengra to find a contraction schedule. 
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

        with progress_reporter.enter_phase("collecting legs"):
            free_legs, leg_indices, index_to_legs = tn._collect_legs()

        open_legs_per_node = defaultdict(list)
        for node_idx, node in tn.nodes.items():
            for leg in node.legs:
                if leg not in free_legs:
                    open_legs_per_node[node_idx].append(_index_leg(node_idx, leg))

        for node_idx, leg_index in open_legs:
            open_legs_per_node[node_idx].append(_index_leg(node_idx, leg_index))

        brute_force_operations = 0
        for node_idx, node in tn.nodes.items():
            h_reduced = gauss(node.h)
            h_reduced = h_reduced[~np.all(h_reduced == 0, axis=1)]
            r = len(h_reduced)
            brute_force_operations += 2 ** r
            
        if cotengra and len(tn.nodes) > 0 and len(tn._traces) > 0:
            with progress_reporter.enter_phase("cotengra contraction"):
                traces, tree = tn._cotengra_contraction(
                    free_legs, leg_indices, index_to_legs, open_legs_per_node, verbose, 
                    progress_reporter, max_repeats=300, minimize=minimize
                )

        tn._traces = traces
        cost_visitor = StabilizerCodeCostVisitor(open_legs_per_node)
        upper_bound_visitor = UpperBoundCostVisitor(open_legs_per_node)

        tn.conjoin_nodes(verbose=verbose, visitors=[cost_visitor, upper_bound_visitor])
        return upper_bound_visitor.total_cost + brute_force_operations, cost_visitor.total_cost + brute_force_operations


def run_contraction_cost_experiment(networks: Dict[Tuple[str, int], TensorNetwork], num_runs, file_name):
    """ Run contraction cost experiments for the given tensor networks and save results to a CSV file.

    Args:
        networks (Dict[Tuple[str, int], TensorNetwork]): A dictionary mapping (name, num_qubits) to tensor networks.
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
    """
    with open(file_name, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["cost_fn", "network", "num_qubits", "num_run", "upper_bound_cost", "contraction_cost"])

    for key, tn in networks.items():
        name, num_qubits = key

        for i in range(num_runs):
            print(f"Finding contraction cost for {name}, run {i+1}")
            upper_bound_cost, custom_cost = find_contraction_cost(tn, minimize="combo", verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

            with open(file_name, "a") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["default", name, num_qubits, i, upper_bound_cost, custom_cost])
        

def run_all_experiments(num_runs=100, file_name="contraction_costs.csv", concatenated=True, rotated_surface=True, rectangular_surface=True, hamming=True, holo=True, bb=True):
    """ Create tensor networks and run contraction cost experiments.
    
    Args:
        num_runs (int): Number of runs for each tensor network.
        file_name (str): Name of the CSV file to save results.
        tn (bool): Whether to include the given tensor network in the experiments.
    """
    tensor_networks = {}
    ds = [3, 5, 7, 9]

    for d in ds:
        # Symmetric Tree Tensor Network (Concatenated Shor's - [[d^2,1,d]]):
        if concatenated:
            tn_concatenated = CompassCodeConcatenateAndSparsifyTN(np.full((d-1, d-1), 2))
            tensor_networks[("Concatenated", d**2)] = tn_concatenated
        
        # Rotated Surface Code - [[d^2,1,d]]
        if rotated_surface:
            compass_code_surface = RotatedSurfaceCodeTN(d)
            tensor_networks[("Rotated Surface", d**2)] = compass_code_surface
        
        # Rectangular Rotated Surface 
        if rectangular_surface:
            coloring = generate_checkerboard_coloring(d, L=3)
            tensor_networks[f"Rectangular Surface {d}"] = CompassCodeRotatedRectangularTN(coloring)

    # MSP for Hamming Code (non-degenerate) -- [[7,1,3]], [[15,7,3]], [[31,21,3]]
    if hamming:
        rs = [3, 4, 5]
        for r in rs:
            tn_hamming = StabilizerMeasurementStatePrepTN(generate_hamming_parity_check(r))
            tensor_networks[("Hamming MSP", 2**r - 1)] = tn_hamming

    # Holographic Happy TN - [[25,11,3]], [[95,51,3]]
    if holo:
        layers = [2, 3, 4]
        for layer in layers:
            tn_holo = HolographicHappyTN(layer)
            tensor_networks[("Holographic", tn_holo.n_qubits())] = tn_holo


    # BB Code MSP
    if bb:
        ls = [3, 3, 6, 12, 12]
        ms = [3, 5, 6, 6, 12]
        a_s = [[1, 0, 2], [0, 1, 2], [3, 1, 2], [3, 1, 2], [3, 2, 7]]
        bs = [[1, 0, 2], [1, 3, 8], [3, 1, 2], [3, 1, 2], [3, 1, 2]]

        # firs two are coprime BB Codes: [[18, 4, 4]] and [[30, 4, 6]] qubit
        for i in range(1,2):
            H_coprime = create_coprime_parity_check(ls[i], ms[i], a_s[i], bs[i])
            tn_bb = StabilizerMeasurementStatePrepTN(H_coprime)
            tensor_networks[("BB MSP", tn_bb.n_qubits())] = tn_bb


        # nest ones are: [[72, 12, 6]], [[144, 12, 12]], [[288, 12, 18]]
        for i in range(2, len(ls)):
            H_reg = create_full_parity_check(ls[i], ms[i], a_s[i], bs[i])
            tn_bb = StabilizerMeasurementStatePrepTN(H_reg)
            tensor_networks[("BB MSP", tn_bb.n_qubits())] = tn_bb

    run_contraction_cost_experiment(tensor_networks, num_runs, file_name)


if __name__ == "__main__":
    run_all_experiments()