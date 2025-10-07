from galois import GF2
from planqtn.legos import Legos
from planqtn.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator
from planqtn.tensor_network import TensorNetwork


class RepCodeTreeConcatenatedTN(TensorNetwork):
    def __init__(self, layers, *, coset_error: GF2 = None, truncate_length: int = None):
        nodes = {}
        connections_to_trace = []
        d = 3
        nodes[(0, 0)] = StabilizerCodeTensorEnumerator(
            Legos.x_rep_code(d + 1), tensor_id=(0, 0)
        )

        for layer in range(1, layers):
            if layer % 2 == 0:
                code = Legos.x_rep_code(d + 1)
            else:
                code = Legos.z_rep_code(d + 1)

            for i in range(d**layer):
                print("d**layer: ", d**layer)
                nodes[(layer, i)] = StabilizerCodeTensorEnumerator(
                    code, tensor_id=(layer, i)
                )
                print("adding node at layer", layer, "index", i)

                # connect this node to its parent in the tree
                parent = (layer - 1, i // d)
                leg = i % d
                connections_to_trace.append((parent, (layer, i), leg, d))

        nodes[(0, 0)] = nodes[(0, 0)].trace_with_stopper(Legos.stopper_i, d)

        super().__init__(nodes, truncate_length=truncate_length)

        for connection in connections_to_trace:
            self.self_trace(
                connection[0], connection[1], [connection[2]], [connection[3]]
            )

        self.n = d**layers
        self.d = d
        self.layers = layers

        self.set_coset(coset_error=coset_error)

    def qubit_to_node_and_leg(self, q):
        L = self.layers - 1

        node_index = q // self.d
        leg_index = q % self.d
        node = (L, node_index)

        return node, leg_index

    def n_qubits(self):
        return self.n
