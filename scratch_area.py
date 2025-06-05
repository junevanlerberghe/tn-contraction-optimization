from collections import defaultdict
import time
from typing import List, Tuple
from galois import GF2
import numpy as np
from qlego.progress_reporter import TqdmProgressReporter
from qlego.simple_poly import SimplePoly
from qlego.progress_reporter import DummyProgressReporter, ProgressReporter
from qlego.parity_check import conjoin, self_trace, sprint, sstr, tensor_product

from qlego.stabilizer_tensor_enumerator import (
    _index_leg,
)

import logging

from compassCodes.compass_code import CompassCode
from qlego.codes.rotated_surface_code import RotatedSurfaceCodeTN
from qlego.tensor_network import _PartiallyTracedEnumerator


#coloring = [[2, 2, 2, 2, 2], [2, 1, 2, 2, 2], [1, 2, 1, 1, 2], [2, 2, 2, 2, 1], [2, 2, 2, 2, 2]]
coloring = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
compass = CompassCode(4, coloring)
tn = compass.rotated()
wep = tn.stabilizer_enumerator_polynomial(
        verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
    )
print(wep)

