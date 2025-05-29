from galois import GF2
import numpy as np
from qlego.compassCodeExperiments.compass_code import CompassCode

class ShorsCode(CompassCode):
    def name(self):
        return "Shor's Code"

    def h_matrix(self):
        return self.dual_surface().conjoin_nodes().h

    def coloring(self):
        return np.full((self.d - 1, self.d - 1), 2)

    def custom_connections(self):
        return [
            ((0, 0), (0, 1), 0, 3),
            ((2, 1), (2, 2), 1, 2),
            ((0, 0), (1, 0), 1, 0),
            ((1, 0), (2, 0), 1, 0),
            ((2, 0), (2, 1), 1, 2),
            ((2, 1), (1, 1), 3, 2),
            ((2, 1), (1, 1), 0, 1),
            ((2, 2), (1, 2), 3, 2),
            ((1, 2), (0, 2), 3, 2),
            ((0, 1), (1, 1), 1, 0),
            ((0, 1), (1, 1), 2, 3),
            ((0, 1), (0, 2), 0, 3),
        ]

class CompassCode1(CompassCode):
    def name(self):
        return "Compass Code 1"

    def h_matrix(self):
        return self.dual_surface().conjoin_nodes().h

    def coloring(self):
        return np.full((self.d - 1, self.d - 1), 2)

    def custom_connections(self):
        return [
            ((0, 0), (0, 1), 0, 3),
            ((2, 1), (2, 2), 1, 2),
            ((0, 0), (1, 0), 1, 0),
            ((1, 0), (2, 0), 1, 0),
            ((2, 0), (2, 1), 1, 2),
            ((2, 1), (1, 1), 3, 2),
            ((2, 1), (1, 1), 0, 1),
            ((2, 2), (1, 2), 3, 2),
            ((1, 2), (0, 2), 3, 2),
            ((0, 1), (1, 1), 1, 0),
            ((0, 1), (1, 1), 2, 3),
            ((0, 1), (0, 2), 0, 3),
        ]