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
        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(PAULI_Z, 0)
            .trace_with_stopper(PAULI_X, 3)
        )
        nodes[(0, last_col)] = (
            nodes[(0, last_col)]
            .trace_with_stopper(PAULI_Z, 2)
            .trace_with_stopper(PAULI_X, 3)
        )
        nodes[(last_row, 0)] = (
            nodes[(last_row, 0)]
            .trace_with_stopper(PAULI_Z, 0)
            .trace_with_stopper(PAULI_X, 1)
        )
        nodes[(last_row, last_col)] = (
            nodes[(last_row, last_col)]
            .trace_with_stopper(PAULI_Z, 2)
            .trace_with_stopper(PAULI_X, 1)
        )

        #TODO: Need to find a way to generalize this, this will just be hardcoded for now
        # go through custom connections to find if there are any legs open
        # top/bottom boundary, if open legs add X stoppers
        # left/right boundary, if open legs add z stoppers
        # corners always remain the same, so leave the above
        nodes[(1, 0)] = (
            nodes[(1, 0)]
            .trace_with_stopper(PAULI_Z, 1)
        )

        nodes[(1, 2)] = (
            nodes[(1, 2)]
            .trace_with_stopper(PAULI_Z, 3)
        )

        nodes[(0,1)] = (
            nodes[(0,1)]
            .trace_with_stopper(PAULI_X, 0)
        )

        nodes[(2,1)] = (
            nodes[(2,1)]
            .trace_with_stopper(PAULI_X, 2)
        )

        super().__init__(nodes, truncate_length=truncate_length)

        for node_a, node_b, leg_a, leg_b in custom_connections:
            self.self_trace(node_a, node_b, [leg_a], [leg_b])

        self.n = d * d
        self.d = d

        self.set_coset(coset_error=coset_error)
    
    def qubit_to_node_and_leg(self, q):
        # col major ordering
        node = (q % self.d, q // self.d)
        return node, (node, 4)

    def n_qubits(self):
        return self.n

