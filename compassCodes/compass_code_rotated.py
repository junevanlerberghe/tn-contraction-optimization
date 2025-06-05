from galois import GF2
import numpy as np
from qlego.legos import Legos
from qlego.tensor_network import PAULI_X, PAULI_Z, TensorNetwork
from qlego.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator


class CompassCodeRotatedTN(TensorNetwork):
    def __init__(
        self,
        d,
        custom_connections: list,
        *,
        lego=lambda node: Legos.enconding_tensor_512,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        """Creates a square compass code based on the connections.

        Uses the rotated surface code layout.
        """
        self.n = d * d
        self.d = d

        nodes = {
            (r, c): StabilizerCodeTensorEnumerator(
                lego((r, c)),
                idx=(r, c),
            )
            # col major ordering
            for r in range(d)
            for c in range(d)
        }

        last_row = d - 1
        last_col = d - 1

        # Apply stoppers to corners
        open_legs = self.find_open_legs(custom_connections)
        top_left_open = open_legs.get((0,0))
        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(PAULI_Z, 2 if 2 in top_left_open else 3)
            .trace_with_stopper(PAULI_X, 3 if 2 in top_left_open else 0)
        )


        top_right_open = open_legs.get((0, last_col))
        nodes[(0, last_col)] = (
            nodes[(0, last_col)]
            .trace_with_stopper(PAULI_Z, 1 if 1 in top_right_open else 0)
            .trace_with_stopper(PAULI_X, 0 if 1 in top_right_open else 3)
        )

        bottom_left_open = open_legs.get((last_row, 0))
        nodes[(last_row, 0)] = (
            nodes[(last_row, 0)]
            .trace_with_stopper(PAULI_Z, 3 if 3 in bottom_left_open else 2)
            .trace_with_stopper(PAULI_X, 2 if 3 in bottom_left_open else 1)
        )

        bottom_right_open = open_legs.get((last_row, last_col))
        nodes[(last_row, last_col)] = (
            nodes[(last_row, last_col)]
            .trace_with_stopper(PAULI_Z, 0 if 0 in bottom_right_open else 1)
            .trace_with_stopper(PAULI_X, 1 if 0 in bottom_right_open else 2)
        )

        # Apply stoppers to sides
        for c in range(1, last_col):
            top_open = open_legs.get((0, c))
            for leg in top_open:
                nodes[(0, c)] = (
                    nodes[(0, c)]
                    .trace_with_stopper(PAULI_X, leg)
                )
            bottom_open = open_legs.get((last_row, c))
            for leg in bottom_open:
                nodes[(last_row, c)] = (
                    nodes[(last_row, c)]
                    .trace_with_stopper(PAULI_X, leg)
                )
        
        for r in range(1, last_row):
            left_open = open_legs.get((r, 0))
            for leg in left_open:
                nodes[(r, 0)] = (
                    nodes[(r, 0)]
                    .trace_with_stopper(PAULI_Z, leg)
                )

            right_open = open_legs.get((r, last_col))
            for leg in right_open:
                nodes[(r, last_col)] = (
                    nodes[(r, last_col)]
                    .trace_with_stopper(PAULI_Z, leg)
                )

        super().__init__(nodes, truncate_length=truncate_length)

        for node_a, node_b, leg_a, leg_b in custom_connections:
            self.self_trace(node_a, node_b, [leg_a], [leg_b])

        self.set_coset(coset_error=coset_error)
    
    def qubit_to_node_and_leg(self, q):
        # col major ordering
        node = (q % self.d, q // self.d)
        return node, (node, 4)

    def n_qubits(self):
        return self.n
    
    def find_open_legs(self, connections):
        used_legs = {}

        for coord1, coord2, leg1, leg2 in connections:
            if coord1 not in used_legs:
                used_legs[coord1] = set()
            if coord2 not in used_legs:
                used_legs[coord2] = set()
            
            used_legs[coord1].add(leg1)
            used_legs[coord2].add(leg2)

        all_legs = set(range(4))
        open_legs = {coord: all_legs - legs for coord, legs in used_legs.items()}
        return open_legs