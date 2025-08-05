"""The `compass_code` module.

It contains the `CompassCodeDualSurfaceCodeLayoutTN` class, which implements a tensor network
representation of compass codes using dual surface code layout.
"""

from typing import Callable, Optional
from galois import GF2
import numpy as np
from planqtn.legos import Legos
from planqtn.networks.surface_code import SurfaceCodeTN
from planqtn.stabilizer_tensor_enumerator import StabilizerCodeTensorEnumerator
from planqtn.tensor_network import TensorId, TensorNetwork


class CompassCodeDualSurfaceCodeLayoutTN(SurfaceCodeTN):
    """A tensor network representation of compass codes using dual surface code layout.

    This class implements a compass code using the dual doubled surface code equivalence
    described by Cao & Lackey in the expansion pack paper. The compass code is constructed
    by applying gauge operations to a surface code based on a coloring pattern.

    Args:
        coloring: Array specifying the coloring pattern for the compass code.
        lego: Function that returns the lego tensor for each node.
        coset_error: Optional coset error for weight enumerator calculations.
        truncate_length: Optional maximum weight for truncating enumerators.
    """

    def __init__(
        self,
        coloring: np.ndarray,
        *,
        lego: Callable[[TensorId], GF2] = lambda node: Legos.encoding_tensor_512,
        coset_error: Optional[GF2] = None,
        truncate_length: Optional[int] = None,
    ):
        """Create a square compass code based on the coloring.

        Creates a compass code using the dual doubled surface code equivalence
        described by Cao & Lackey in the expansion pack paper.

        Args:
            coloring: Array specifying the coloring pattern for the compass code.
            lego: Function that returns the lego tensor for each node.
            coset_error: Optional coset error for weight enumerator calculations.
            truncate_length: Optional maximum weight for truncating enumerators.
        """
        # See d3_compass_code_numbering.png for numbering - for an (r,c) qubit in the compass code,
        # the (2r, 2c) is the coordinate of the lego in the dual surface code.
        d = len(coloring) + 1
        super().__init__(d=d, lego=lego, truncate_length=truncate_length)
        gauge_idxs = [
            (r, c) for r in range(1, 2 * d - 1, 2) for c in range(1, 2 * d - 1, 2)
        ]
        for tensor_id, color in zip(gauge_idxs, np.reshape(coloring, (d - 1) ** 2)):
            self.nodes[tensor_id] = self.nodes[tensor_id].trace_with_stopper(
                Legos.stopper_z if color == 2 else Legos.stopper_x, 4
            )

        self._q_to_node = [(2 * r, 2 * c) for c in range(d) for r in range(d)]
        self.n = d * d
        self.coloring = coloring

        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )



class CompassCodeRotatedTN(TensorNetwork):
    def __init__(
        self,
        coloring: np.ndarray,
        *,
        lego=lambda node: Legos.encoding_tensor_512,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        """Creates a square compass code based on the connections.

        Uses the rotated surface code layout.
        """
        d = len(coloring) + 1
        self.d = d
        self.n = d * d
        self.coloring = coloring
        custom_connections = self.make_custom_connections()

        nodes = {
            (r, c): StabilizerCodeTensorEnumerator(
                lego((r, c)),
                tensor_id=(r, c),
            )
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
            .trace_with_stopper(Legos.stopper_z, 2 if 2 in top_left_open else 3)
            .trace_with_stopper(Legos.stopper_x, 3 if 2 in top_left_open else 0)
        )

        top_right_open = open_legs.get((0, last_col))
        nodes[(0, last_col)] = (
            nodes[(0, last_col)]
            .trace_with_stopper(Legos.stopper_z, 1 if 1 in top_right_open else 0)
            .trace_with_stopper(Legos.stopper_x, 0 if 1 in top_right_open else 3)
        )

        bottom_left_open = open_legs.get((last_row, 0))
        nodes[(last_row, 0)] = (
            nodes[(last_row, 0)]
            .trace_with_stopper(Legos.stopper_z, 3 if 3 in bottom_left_open else 2)
            .trace_with_stopper(Legos.stopper_x, 2 if 3 in bottom_left_open else 1)
        )

        bottom_right_open = open_legs.get((last_row, last_col))
        nodes[(last_row, last_col)] = (
            nodes[(last_row, last_col)]
            .trace_with_stopper(Legos.stopper_z, 0 if 0 in bottom_right_open else 1)
            .trace_with_stopper(Legos.stopper_x, 1 if 0 in bottom_right_open else 2)
        )

        # Apply stoppers to sides
        for c in range(1, last_col):
            top_open = open_legs.get((0, c))
            for leg in top_open:
                nodes[(0, c)] = (
                    nodes[(0, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
            bottom_open = open_legs.get((last_row, c))
            for leg in bottom_open:
                nodes[(last_row, c)] = (
                    nodes[(last_row, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
        
        for r in range(1, last_row):
            left_open = open_legs.get((r, 0))
            for leg in left_open:
                nodes[(r, 0)] = (
                    nodes[(r, 0)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )

            right_open = open_legs.get((r, last_col))
            for leg in right_open:
                nodes[(r, last_col)] = (
                    nodes[(r, last_col)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )

        super().__init__(nodes, truncate_length=truncate_length)

        for node_a, node_b, leg_a, leg_b in custom_connections:
            self.self_trace(node_a, node_b, [leg_a], [leg_b])

        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )

    def make_custom_connections(self):
        connections = set()
        # start with connections for Shor's code
        for row in range(self.d):
            for col in range(self.d):
                if(col < self.d - 1):
                    if(row == 0):
                        connections.add(((row, col), (row, col + 1), 0, 3))
                    if(row == self.d - 1):
                        connections.add(((row, col), (row, col + 1), 1, 2))
                if(row < self.d - 1):
                    if(col < self.d - 1):
                        connections.add(((row, col), (row + 1, col), 1, 0))
                    if(col > 0):
                        connections.add(((row, col), (row + 1, col), 2, 3))
        
        # now change connections to adjust for the specific coloring
        for row in range(self.d - 1):
            for col in range(self.d - 1):
                if(self.coloring[row][col] == 1):
                    top_left_qubit = (row, col)
                    top_right_qubit = (row, col + 1)
                    bottom_left_qubit = (row + 1, col)
                    bottom_right_qubit = (row + 1, col + 1)

                    connections.remove((top_left_qubit, bottom_left_qubit, 1, 0))
                    connections.remove((top_right_qubit, bottom_right_qubit, 2, 3))

                    if(col == self.d - 2):
                        connections.add((top_right_qubit, bottom_right_qubit, 1, 0))
                    if(col == 0):
                        connections.add((top_left_qubit, bottom_left_qubit, 2, 3))
                    if(row == 0):
                        connections.remove((top_left_qubit, top_right_qubit, 0, 3))
                    if(row == self.d - 2):
                        connections.remove((bottom_left_qubit, bottom_right_qubit, 1, 2))

                    connections.add((top_left_qubit, top_right_qubit, 1, 2))
                    connections.add((bottom_left_qubit, bottom_right_qubit, 0, 3))

        return connections
    
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
    

class CompassCodeConcatenateAndSparsifyTN(TensorNetwork):
    def __init__(
        self,
        coloring,
        *,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        d = len(coloring) + 1
        nodes = {}
        attachments = {}
        nodes[(0,0)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(d+1), tensor_id=(0,0))
            
        for c in range(d):
            nodes[(1,c)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(d+1), tensor_id=(1,c))
            for leg in range(d):
                attachments[(leg,c)] = ((1, c), leg)

        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(Legos.stopper_i, d)
        )

        connections_to_trace = set()
        trace_with_stopper = set()

        for col in range(len(coloring[0])):
            # Skip this column if there are no 1s in it
            if not any(coloring[row][col] == 1 for row in range(len(coloring))):
                continue
            
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

                    gap_above = max(0, start_row - (last_zero_row + 1))  
                    gap_below = max(0, next_one - end_row - 1)
                    if gap_above <= gap_below:
                        # Merge upward (use rows from start_row to end_row)
                        z_merge_key = ("z_merge", start_row, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(block_size), tensor_id=z_merge_key)

                        for offset, j in enumerate(range(start_row, end_row + 1)):

                            nodes[("x1", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x1", j, col))
                            nodes[("z", j, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), tensor_id=("z", j, col))
                            nodes[("x2", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x2", j, col))

                            connections_to_trace.add((("x1", j, col), ("z", j, col), 0, 1))
                            connections_to_trace.add((("z", j, col), ("x2", j, col), 0, 1))

                            qubit1, leg1 = attachments[(j, col)]
                            qubit2, leg2 = attachments[(j, col + 1)]
                            print("\t adding non-isometry between qubits", qubit1, "and", qubit2, "at row", j, "col", col)
                            connections_to_trace.add((qubit1, ("x1", j, col), leg1, 2))
                            connections_to_trace.add((qubit2, ("x2", j, col), leg2, 2))

                            attachments[(j, col)] = (("x1", j, col), 1)
                            attachments[(j, col + 1)] = (("x2", j, col), 0)

                            connections_to_trace.add((("z", j, col), z_merge_key, 2, offset))

                    else:
                        extra_rows = next_one - (end_row + 1)
                        z_merge_key = ("z_merge", end_row + 1, col)
                        nodes[z_merge_key] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(extra_rows), tensor_id=z_merge_key)

                        for offset, j in enumerate(range(end_row + 1, next_one)):
                            nodes[("x1", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x1", j, col))
                            nodes[("z", j, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), tensor_id=("z", j, col))
                            nodes[("x2", j, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x2", j, col))

                            connections_to_trace.add((("x1", j, col), ("z", j, col), 0, 1))
                            connections_to_trace.add((("z", j, col), ("x2", j, col), 0, 1))

                            qubit1, leg1 = attachments[(j, col)]
                            qubit2, leg2 = attachments[(j, col + 1)]
                            print("\t adding non-isometry between qubits", qubit1, "and", qubit2, "at row", j, "col", col)
                            connections_to_trace.add((qubit1, ("x1", j, col), leg1, 2))
                            connections_to_trace.add((qubit2, ("x2", j, col), leg2, 2))

                            attachments[(j, col)] = (("x1", j, col), 1)
                            attachments[(j, col + 1)] = (("x2", j, col), 0)

                            connections_to_trace.add((("z", j, col), z_merge_key, 2, offset))

                        row = next_one - 1  

                row += 1 

            top_rows = []
            bottom_rows = []
            height = coloring.shape[0]

            # --- Find contiguous top block of 1s ---
            row = 0
            while row < height and coloring[row][col] == 1:
                top_rows.append(row)
                row += 1
     
            # --- Find contiguous bottom block of 1s ---
            row = height - 1
            while row >= 0 and coloring[row][col] == 1:
                bottom_rows.append(row + 1)
                row -= 1

            bottom_rows = list(reversed(bottom_rows))  # ensure increasing order

            # --- Avoid duplication if full column is 1s ---
            full_column_ones = len(top_rows) + len(bottom_rows) > height
            if full_column_ones:
                # Only apply from the top to avoid duplication
                bottom_rows = []
                if(len(top_rows) > 1):
                    top_rows.append(top_rows[-1] + 1)

            # --- Apply non-isometry at top rows ---
            for r in top_rows:
                print(f"adding non-isometry at col {col}, row {r} (top)")
                nodes[("x1", r, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x1", r, col))
                nodes[("z", r, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), tensor_id=("z", r, col))
                nodes[("x2", r, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x2", r, col))

                connections_to_trace.add((("x1", r, col), ("z", r, col), 0, 1))
                connections_to_trace.add((("z", r, col), ("x2", r, col), 0, 1))

                qubit1, leg1 = attachments[(r, col)]
                qubit2, leg2 = attachments[(r, col + 1)]
                print("\t qubit1: ", qubit1, " leg1: ", leg1)
                print("\t qubit2: ", qubit2, " leg2: ", leg2)
                connections_to_trace.add((qubit1, ("x1", r, col), leg1, 2))
                connections_to_trace.add((qubit2, ("x2", r, col), leg2, 2))

                attachments[(r, col)] = (("x1", r, col), 1)
                attachments[(r, col + 1)] = (("x2", r, col), 0)

                trace_with_stopper.add(("z", r, col))

            # --- Apply non-isometry at bottom rows ---
            for r in bottom_rows:
                print(f"adding non-isometry at col {col}, row {r} (bottom)")
                nodes[("x1", r, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x1", r, col))
                nodes[("z", r, col)] = StabilizerCodeTensorEnumerator(Legos.z_rep_code(3), tensor_id=("z", r, col))
                nodes[("x2", r, col)] = StabilizerCodeTensorEnumerator(Legos.x_rep_code(3), tensor_id=("x2", r, col))

                connections_to_trace.add((("x1", r, col), ("z", r, col), 0, 1))
                connections_to_trace.add((("z", r, col), ("x2", r, col), 0, 1))

                qubit1, leg1 = attachments[(r, col)]
                qubit2, leg2 = attachments[(r, col + 1)]

                connections_to_trace.add((qubit1, ("x1", r, col), leg1, 2))
                connections_to_trace.add((qubit2, ("x2", r, col), leg2, 2))

                attachments[(r, col)] = (("x1", r, col), 1)
                attachments[(r, col + 1)] = (("x2", r, col), 0)

                trace_with_stopper.add(("z", r, col))

        
        super().__init__(nodes, truncate_length=truncate_length)

        for leg in range(d):
            self.self_trace((0,0), (1,leg), [leg], [d])
 
        for connection in connections_to_trace:
            self.self_trace(connection[0], connection[1], [connection[2]], [connection[3]])  

        for node in trace_with_stopper:
            self.nodes[node] = (
                self.nodes[node]
                .trace_with_stopper(Legos.stopper_x, 2)
            )      

        self.n = d * d
        self.d = d
        print(attachments)
        self.attachments = attachments
        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )

    def qubit_to_node_and_leg(self, q):
        idx_leg = q % self.d
        idx_node = q // self.d
        node, leg = self.attachments[(idx_leg, idx_node)]
        return node, (node, leg)

    def n_qubits(self):
        return self.n



class CompassCodeRotatedRectangularTN(TensorNetwork):
    def __init__(
        self,
        coloring: np.ndarray,
        *,
        lego=lambda node: Legos.encoding_tensor_512,
        coset_error: GF2 = None,
        truncate_length: int = None
    ):
        """Creates a square compass code based on the connections.

        Uses the rotated surface code layout.
        """
        coloring = np.array(coloring)
        n = (coloring.shape[0] + 1) * (coloring.shape[1] + 1)
        L = coloring.shape[0] + 1
        self.d = n//L
        self.n = n
        self.L = L
        self.coloring = coloring
        custom_connections = self.make_custom_connections()

        nodes = {
            (r, c): StabilizerCodeTensorEnumerator(
                lego((r, c)),
                tensor_id=(r, c),
            )
            for r in range(L)
            for c in range(n//L)
        }

        last_row = L - 1
        last_col = n//L - 1

        # Apply stoppers to corners
        open_legs = self.find_open_legs(custom_connections)
        top_left_open = open_legs.get((0,0))
        nodes[(0, 0)] = (
            nodes[(0, 0)]
            .trace_with_stopper(Legos.stopper_z, 2 if 2 in top_left_open else 3)
            .trace_with_stopper(Legos.stopper_x, 3 if 2 in top_left_open else 0)
        )

        top_right_open = open_legs.get((0, last_col))
        nodes[(0, last_col)] = (
            nodes[(0, last_col)]
            .trace_with_stopper(Legos.stopper_z, 1 if 1 in top_right_open else 0)
            .trace_with_stopper(Legos.stopper_x, 0 if 1 in top_right_open else 3)
        )

        bottom_left_open = open_legs.get((last_row, 0))
        nodes[(last_row, 0)] = (
            nodes[(last_row, 0)]
            .trace_with_stopper(Legos.stopper_z, 3 if 3 in bottom_left_open else 2)
            .trace_with_stopper(Legos.stopper_x, 2 if 3 in bottom_left_open else 1)
        )

        bottom_right_open = open_legs.get((last_row, last_col))
        nodes[(last_row, last_col)] = (
            nodes[(last_row, last_col)]
            .trace_with_stopper(Legos.stopper_z, 0 if 0 in bottom_right_open else 1)
            .trace_with_stopper(Legos.stopper_x, 1 if 0 in bottom_right_open else 2)
        )

        # Apply stoppers to sides
        for c in range(1, last_col):
            top_open = open_legs.get((0, c))
            for leg in top_open:
                nodes[(0, c)] = (
                    nodes[(0, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
            bottom_open = open_legs.get((last_row, c))
            for leg in bottom_open:
                nodes[(last_row, c)] = (
                    nodes[(last_row, c)]
                    .trace_with_stopper(Legos.stopper_x, leg)
                )
        
        for r in range(1, last_row):
            left_open = open_legs.get((r, 0))
            for leg in left_open:
                nodes[(r, 0)] = (
                    nodes[(r, 0)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )

            right_open = open_legs.get((r, last_col))
            for leg in right_open:
                nodes[(r, last_col)] = (
                    nodes[(r, last_col)]
                    .trace_with_stopper(Legos.stopper_z, leg)
                )

        super().__init__(nodes, truncate_length=truncate_length)

        for node_a, node_b, leg_a, leg_b in custom_connections:
            self.self_trace(node_a, node_b, [leg_a], [leg_b])

        self.set_coset(
            coset_error if coset_error is not None else GF2.Zeros(2 * self.n)
        )

    def make_custom_connections(self):
        connections = set()
        # start with connections for Shor's code
        for row in range(self.L):
            for col in range(self.d):
                if(col < self.d - 1):
                    if(row == 0):
                        connections.add(((row, col), (row, col + 1), 0, 3))
                    if(row == self.L - 1):
                        connections.add(((row, col), (row, col + 1), 1, 2))
                if(row < self.L - 1):
                    if(col < self.d - 1):
                        connections.add(((row, col), (row + 1, col), 1, 0))
                    if(col > 0):
                        connections.add(((row, col), (row + 1, col), 2, 3))
        
        # now change connections to adjust for the specific coloring
        for row in range(self.L - 1):
            for col in range(self.d- 1):
                if(self.coloring[row][col] == 1):
                    top_left_qubit = (row, col)
                    top_right_qubit = (row, col + 1)
                    bottom_left_qubit = (row + 1, col)
                    bottom_right_qubit = (row + 1, col + 1)

                    connections.remove((top_left_qubit, bottom_left_qubit, 1, 0))
                    connections.remove((top_right_qubit, bottom_right_qubit, 2, 3))

                    if(col == self.d - 2):
                        connections.add((top_right_qubit, bottom_right_qubit, 1, 0))
                    if(col == 0):
                        connections.add((top_left_qubit, bottom_left_qubit, 2, 3))
                    if(row == 0):
                        connections.remove((top_left_qubit, top_right_qubit, 0, 3))
                    if(row == self.L - 2):
                        connections.remove((bottom_left_qubit, bottom_right_qubit, 1, 2))

                    connections.add((top_left_qubit, top_right_qubit, 1, 2))
                    connections.add((bottom_left_qubit, bottom_right_qubit, 0, 3))

        return connections
    
    def qubit_to_node_and_leg(self, q):
        # col major ordering
        node = (q % self.L, q // self.L)
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