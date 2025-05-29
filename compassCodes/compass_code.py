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
            "Tanner Network": self.tanner,
            "Measurement State Prep": self.msp,
            "Dual Surface": self.dual_surface,
            "Concatenated": self.concatenated,
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
        # given coloring, make custom connections
        # start from surface code and make adjustments
        surface_connections = set()
        for radius in range(1, self.d):
            for i in range(radius + 1):
                surface_connections.add(((i, radius - 1),
                    (i, radius),
                    3 if (i + radius) % 2 == 0 else 2,
                    0 if (i + radius) % 2 == 0 else 1,
                ))
                if i > 0 and i < radius:
                    surface_connections.add((
                        (i - 1, radius),
                        (i, radius),
                        2 if (i + radius) % 2 == 0 else 1,
                        3 if (i + radius) % 2 == 0 else 0,
                    ))

                surface_connections.add((
                    (radius - 1, i),
                    (radius, i),
                    2 if (i + radius) % 2 == 0 else 1,
                    3 if (i + radius) % 2 == 0 else 0,
                ))
                if i > 0 and i < radius:
                    surface_connections.add((
                        (radius, i - 1),
                        (radius, i),
                        3 if (i + radius) % 2 == 0 else 2,
                        0 if (i + radius) % 2 == 0 else 1,
                    ))
        # go through coloring to see what needs to be changed 
        # if anything that's supposed to be an X is a 2, then change
        # loop through coloring by 2s to loop through only Xs, if anything is a 2, then change

        for row in range(self.d - 1):
            start_col = 1 if row % 2 == 0 else 0
            for col in range(start_col, self.d - 1, 2):
                if self.coloring[row][col] == 2:
                    print("twisting qubits around coord: ", row, col)
                    # we have row and col of the stabilizer that needs to be changed
                    top_left = (row, col)
                    top_right = (row, col + 1)
                    bottom_left = (row + 1, col)
                    bottom_right = (row+1, col+1)

                    hor_conn_top = (top_left, 
                                    top_right, 
                                    2 if (row + col) % 2 == 0 else 3,
                                    1 if (row + col) % 2 == 0 else 0)
                    hor_conn_bottom = (bottom_left, 
                                       bottom_right,
                                    3 if (row + col) % 2 == 0 else 2,
                                    0 if (row + col) % 2 == 0 else 1)

                    if(row != 0):
                        print("removing top connection: " , hor_conn_top)
                        surface_connections.remove(hor_conn_top)
                    if(row != self.d - 2):
                        print("removing bottom connection: " , hor_conn_bottom)
                        surface_connections.remove(hor_conn_bottom)

                    vert_conn_left = (top_left, 
                                      bottom_left, 
                                      1 if (row + col) % 2 == 0 else 2, 
                                      0 if (row + col) % 2 == 0 else 3)
                    vert_conn_right = (top_right, 
                                       bottom_right, 
                                       2 if (row + col) % 2 == 0 else 1, 
                                       3 if (row + col) % 2 == 0 else 0)
                    if(col != 0):
                        print("adding left connection: " , vert_conn_left)
                        surface_connections.add(vert_conn_left)
                    if(col != self.d - 2):
                        print("adding right connection: ", vert_conn_right)
                        surface_connections.add(vert_conn_right)

        return surface_connections
        
        # stoppers will be adjusted in the compasscodetn itself based on these connections
        # might still have to figure out legs somehow