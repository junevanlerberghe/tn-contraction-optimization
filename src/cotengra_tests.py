import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "planqtn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compassCodes.compass_code_concatenated import RepCodeTreeConcatenatedTN
from cotengra import OptimalOptimizer
from cotengra.pathfinders.path_basic import compute_size_custom
from planqtn.contraction_visitors.stabilizer_flops_cost_fn import StabilizerCodeFlopsCostVisitor
from planqtn.networks.rotated_surface_code import RotatedSurfaceCodeTN
from planqtn.tensor_network import Contraction

def test_custom_optimal_optimizer_rotated():
    tn = RotatedSurfaceCodeTN(d=3)
    contraction = Contraction(
        tn,
        lambda node: node.copy(),
    )
    contraction_copy = Contraction(
        tn,
        lambda node: node.copy(),
    )
    opt = OptimalOptimizer(minimize="custom_flops", contraction_info=contraction_copy)
    search_params = {"optimal_minimizer": "custom_flops", "contraction_info": contraction_copy}
    contraction._cot_tree = opt.search(contraction.inputs, contraction.output, contraction.size_dict, search_params)

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    contraction.contract(visitors=[cost_visitor], cotengra=False)
    assert cost_visitor.total_cost == 68


def test_custom_optimal_optimizer_rep_code():
    tn = RepCodeTreeConcatenatedTN(2)
    contraction = Contraction(
        tn,
        lambda node: node.copy(),
    )
    contraction_copy = Contraction(
        tn,
        lambda node: node.copy(),
    )
    opt = OptimalOptimizer(minimize="custom_flops", contraction_info=contraction_copy)
    search_params = {"optimal_minimizer": "custom_flops", "contraction_info": contraction_copy}
    contraction._cot_tree = opt.search(contraction.inputs, contraction.output, contraction.size_dict, search_params)

    cost_visitor = StabilizerCodeFlopsCostVisitor()
    contraction.contract(visitors=[cost_visitor], cotengra=False)
    assert cost_visitor.total_cost == 10
    

def test_greedy_custom_size():
    tn = RotatedSurfaceCodeTN(d=3)
    contraction = Contraction(
        tn,
        lambda node: node.copy(),
    )

    combined_traces = [[((0, 0), (0, 1), ((0, 0), 2), ((0, 1), 1))]]
    new_size = compute_size_custom(contraction, combined_traces)
    assert new_size == 4

    combined_traces = [[((0, 0), (0, 1), ((0, 0), 2), ((0, 1), 1))], [((0, 0), (1, 0), ((0, 0), 1), ((1, 0), 0))]]
    new_size = compute_size_custom(contraction, combined_traces)
    assert new_size == 16

    combined_traces = [[((0, 0), (0, 1), ((0, 0), 2), ((0, 1), 1))],
                       [((0, 0), (1, 0), ((0, 0), 1), ((1, 0), 0))], 
                       [((0, 1), (1, 1), ((0, 1), 2), ((1, 1), 3)), 
                       ((1, 0), (1, 1), ((1, 0), 3), ((1, 1), 0))], 
                       [((0, 1), (0, 2), ((0, 1), 3), ((0, 2), 0))]]
    new_size = compute_size_custom(contraction, combined_traces)
    assert new_size == 32