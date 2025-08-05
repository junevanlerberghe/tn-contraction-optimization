# MSP for Hamming Code (non-degenerate)
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tnqec'))

from planqtn.networks.stabilizer_measurement_state_prep import StabilizerMeasurementStatePrepTN
from planqtn.progress_reporter import TqdmProgressReporter
from upper_bound_info import run_upper_bound_calc


h_x_hamming = [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
]
top = np.hstack((h_x_hamming, np.zeros([len(h_x_hamming), len(h_x_hamming[0])])))
bottom = np.hstack((np.zeros([len(h_x_hamming), len(h_x_hamming[0])]), h_x_hamming))
H_hamming = np.vstack((top, bottom))
print(H_hamming)
tn_hamming = StabilizerMeasurementStatePrepTN(H_hamming)

legs_cost, custom_cost = run_upper_bound_calc(tn_hamming, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)

wep = tn_hamming.stabilizer_enumerator_polynomial()
print(wep)