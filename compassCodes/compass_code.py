from qlego.codes.stabilizer_tanner_code import StabilizerTannerCodeTN
from qlego.legos import Legos
from qlego.codes.stabilizer_measurement_state_prep import StabilizerMeasurementStatePrepTN
from compassCodes.compass_code_rotated import CompassCodeRotatedTN
from compassCodes.compass_code_concatenated import CompassCodeConcatenatedTN
from qlego.codes.compass_code import CompassCodeTN


class CompassCode():
    def __init__(self, d: int, coloring):
        self.d = d
        self.coloring = coloring
        self.representations = {
            "Dual Surface": self.dual_surface,
            "Rotated Surface": self.rotated,
            "Concatenated": self.concatenated,
            "Measurement State Prep": self.msp,
            "Tanner Network": self.tanner
        }

    def h_matrix(self):
        return self.dual_surface().conjoin_nodes().h

    def coloring(self):
        return self.coloring

    def custom_connections(self):
        """Returns list of CompassCodeRotatedTN connections"""
        pass

    # ===== Representation Implementations =====
    def tanner(self):
        return StabilizerTannerCodeTN(self.h_matrix())

    def msp(self):
        return StabilizerMeasurementStatePrepTN(self.h_matrix())

    def dual_surface(self):
        return CompassCodeTN(self.coloring, lego=lambda i: Legos.enconding_tensor_512)

    def rotated(self):
        return CompassCodeRotatedTN(d=self.d, custom_connections=self.make_custom_connections())

    def concatenated(self):
        return CompassCodeConcatenatedTN(self.d, coloring=self.coloring)
    
    def get_representations(self):
        return self.representations
    
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