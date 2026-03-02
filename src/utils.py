import itertools
import numpy as np


def generate_checkerboard_coloring(d, L=None):
    if L is None:
        L = d
    return np.array([[1 + (i + j) % 2 for j in range(L - 1)] for i in range(d - 1)])


def generate_hamming_hx(r):
    bitstrings = [
        np.array(bits) for bits in itertools.product([0, 1], repeat=r) if any(bits)
    ]
    H = np.array(bitstrings).T
    return H


def generate_hamming_parity_check(r):
    """Generates the full parity check matrix for the Hamming code of order r."""
    H_x = generate_hamming_hx(r)
    top = np.hstack((H_x, np.zeros([len(H_x), len(H_x[0])])))
    bottom = np.hstack((np.zeros([len(H_x), len(H_x[0])]), H_x))
    H_hamming = np.vstack((top, bottom))
    return H_hamming

def generate_rotated_surface_code(d):
    """
    Generates the symplectic parity check matrix for a rotated surface code
    of distance d.
    
    Qubit Ordering: Column-major
    0  3  6
    1  4  7
    2  5  8
    
    Output Format:
    Symplectic matrix H = [Hx | Hz]
    Rows are stabilizers, Columns are 2 * n_qubits.
    """
    n_qubits = d**2
    
    qubit_map = {}
    for c in range(d):
        for r in range(d):
            qubit_map[(r, c)] = c * d + r

    stabilizers_x = []
    stabilizers_z = []

    for c in range(-1, d):
        for r in range(-1, d):
            
            # Determine Check Type based on lattice parity
            is_x_check = ((r + c) % 2 == 0)
            
            potential_qubits = [
                (r, c),     (r+1, c),
                (r, c+1),   (r+1, c+1)
            ]
            
            # Filter for valid qubits that exist on the d x d grid
            valid_qubit_indices = []
            for q_coord in potential_qubits:
                if q_coord in qubit_map:
                    valid_qubit_indices.append(qubit_map[q_coord])
            
            if not valid_qubit_indices:
                continue

            # Weight 4 checks are always kept 
            # Weight 2 checks are kept only on specific boundaries
            keep_check = False
            
            if len(valid_qubit_indices) == 4:
                keep_check = True
            elif len(valid_qubit_indices) == 2:
                if is_x_check:
                    # X-Checks allowed on Left (c=-1) and Right (c=d-1) edges
                    if c == -1 or c == d - 1:
                        keep_check = True
                else:
                    # Z-Checks allowed on Top (r=-1) and Bottom (r=d-1) edges
                    if r == -1 or r == d - 1:
                        keep_check = True

            # 4. Construct the Symplectic Row
            if keep_check:
                row = np.zeros(2 * n_qubits, dtype=int)
                
                # If X check, place 1s in first half (0 to N-1)
                # If Z check, place 1s in second half (N to 2N-1)
                offset = 0 if is_x_check else n_qubits
                
                for q_idx in valid_qubit_indices:
                    row[offset + q_idx] = 1
                
                if is_x_check:
                    stabilizers_x.append(row)
                else:
                    stabilizers_z.append(row)

    # Stack X stabilizers on top of Z stabilizers
    if stabilizers_x and stabilizers_z:
        H = np.vstack(stabilizers_x + stabilizers_z)
    else:
        H = np.array([])

    return H