import itertools
import numpy as np

def generate_checkerboard_coloring(d, L=None):
    if(L is None):
        L = d
    return np.array([[1 + (i + j) % 2 for j in range(L-1)] for i in range(d-1)])

def generate_hamming_hx(r):
    bitstrings = [np.array(bits) for bits in itertools.product([0, 1], repeat=r) if any(bits)]
    H = np.array(bitstrings).T
    return H

def generate_hamming_parity_check(r):
    """Generates the full parity check matrix for the Hamming code of order r."""
    H_x = generate_hamming_hx(r)
    top = np.hstack((H_x, np.zeros([len(H_x), len(H_x[0])])))
    bottom = np.hstack((np.zeros([len(H_x), len(H_x[0])]), H_x))
    H_hamming = np.vstack((top, bottom))
    return H_hamming
