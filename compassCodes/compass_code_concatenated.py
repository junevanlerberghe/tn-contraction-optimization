from galois import GF2
import numpy as np
from qlego.legos import Legos
from qlego.tensor_network import PAULI_X, PAULI_Z, PAULI_I, TensorNetwork
from qlego.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator


class CompassCodeConcatenatedTN(TensorNetwork):
    def __init__(
        self,
        d,
        coloring,
        *,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        nodes = {}
        attachments = {}
        nodes[(0,0)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(d+1), idx=(0,0))
            
        for c in range(d):
            nodes[(1,c)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(d+1), idx=(1,c))
            for leg in range(d):
                attachments[(leg,c)] = ((1, c), leg)

        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(PAULI_I, d)
        )

        connections_to_trace = set()
        trace_with_stopper = set()
        for col in range(len(coloring[0])):
            row = 0
            while row < len(coloring):
                if coloring[row][col] == 2:
                    start_row = row
                    while row + 1 < len(coloring) and coloring[row + 1][col] == 2:
                        row += 1
                    end_row = row + 1
                    block_size = end_row - start_row + 1
                    last_zero_row = start_row - 1
                    next_one = next((r for r in range(end_row + 1, len(coloring)) if coloring[r][col] == 1), len(coloring) + 1)
                    print("next one is: ", next_one)
                    gap_above = max(0, start_row - (last_zero_row + 1))  
                    gap_below = max(0, next_one - end_row - 1)
                    if gap_above <= gap_below:
                        # Merge upward (use rows from start_row to end_row)
                        z_merge_key = ("z_merge", start_row, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(block_size), idx=z_merge_key)

                        print("looping to apply non-isometries at col", col, "from row", start_row, "to", end_row+1)
                        for offset, j in enumerate(range(start_row, end_row + 1)):
                            nodes[("x1", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x1", j, col))
                            nodes[("z", j, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), idx=("z", j, col))
                            nodes[("x2", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x2", j, col))

                            connections_to_trace.add((("x1", j, col), ("z", j, col), 0, 1))
                            connections_to_trace.add((("z", j, col), ("x2", j, col), 0, 1))

                            qubit1, leg1 = attachments[(j, col)]
                            qubit2, leg2 = attachments[(j, col + 1)]
                            print("\t adding non-isometry between qubits", qubit1, "and", qubit2, "at row", j, "col", col)
                            print("\t on legs", leg1, "and", leg2)
                            connections_to_trace.add((qubit1, ("x1", j, col), leg1, 2))
                            connections_to_trace.add((qubit2, ("x2", j, col), leg2, 2))

                            attachments[(j, col)] = (("x1", j, col), 1)
                            attachments[(j, col + 1)] = (("x2", j, col), 0)

                            connections_to_trace.add((("z", j, col), z_merge_key, 2, offset))

                    else:
                        extra_rows = next_one - (end_row + 1)
                        z_merge_key = ("z_merge", end_row + 1, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(extra_rows), idx=z_merge_key)

                        print("looping to apply non-isometries at col", col, "from row", end_row + 1, "to", next_one)
                        for offset, j in enumerate(range(end_row + 1, next_one)):
                            nodes[("x1", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x1", j, col))
                            nodes[("z", j, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), idx=("z", j, col))
                            nodes[("x2", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x2", j, col))

                            connections_to_trace.add((("x1", j, col), ("z", j, col), 0, 1))
                            connections_to_trace.add((("z", j, col), ("x2", j, col), 0, 1))

                            qubit1, leg1 = attachments[(j, col)]
                            qubit2, leg2 = attachments[(j, col + 1)]
                            print("\t adding non-isometry between qubits", qubit1, "and", qubit2, "at row", j, "col", col)
                            print("\t on legs", leg1, "and", leg2)
                            connections_to_trace.add((qubit1, ("x1", j, col), leg1, 2))
                            connections_to_trace.add((qubit2, ("x2", j, col), leg2, 2))

                            attachments[(j, col)] = (("x1", j, col), 1)
                            attachments[(j, col + 1)] = (("x2", j, col), 0)

                            connections_to_trace.add((("z", j, col), z_merge_key, 2, offset))

                        row = next_one - 1  

                row += 1 

            if(coloring[0][col] == 1 and coloring[-1][col] == 1):
                print("adding non-isometry at col", col, "at top")
                nodes[("x1", 0, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x1", 0, col))
                nodes[("z", 0, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), idx=("z", 0, col))
                nodes[("x2", 0, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), idx=("x2", 0, col))

                connections_to_trace.add((("x1", 0, col), ("z", 0, col), 0, 1))
                connections_to_trace.add((("z", 0, col), ("x2", 0, col), 0, 1))

                qubit1, leg1 = attachments[(0, col)]
                qubit2, leg2 = attachments[(0, col + 1)]
                print("\t adding single non-isometry between qubits", qubit1, "and", qubit2, "at row", 0, "col", col)
                print("\t on legs", leg1, "and", leg2)
                connections_to_trace.add((qubit1, ("x1", 0, col), leg1, 2))
                connections_to_trace.add((qubit2, ("x2", 0, col), leg2, 2))

                attachments[(1, col)] = (("x1", 0, col), 1)
                attachments[(0, col + 1)] = (("x2", 0, col), 0)

                # want to trace the Z node of non-isometry
                trace_with_stopper.add(("z", 0, col))        
        
        super().__init__(nodes, truncate_length=truncate_length)

        for leg in range(d):
            print("self tracing qubits:", (0,0), (1,leg), [leg], [d])
            self.self_trace((0,0), (1,leg), [leg], [d])
 
        for connection in connections_to_trace:
            print("self tracing connection:", connection)
            self.self_trace(connection[0], connection[1], [connection[2]], [connection[3]])  

        for node in trace_with_stopper:
            print("self tracing node:", node)
            self.nodes[node] = (
                self.nodes[node]
                .trace_with_stopper(PAULI_X, 2)
            )      

        self.n = d * d
        self.d = d

        self.set_coset(coset_error=coset_error)

    def qubit_to_node_and_leg(self, q):
        for c in range(self.d):
            for leg in range(self.d):
                if(q == self.d*c + leg):
                    node = (1,c)
                    return node, (node, leg)

    def n_qubits(self):
        return self.n

