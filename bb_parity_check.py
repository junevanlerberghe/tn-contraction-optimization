from typing import List
import numpy as np

def make_cyclic_shift_matrix(l: int) -> np.array:
    result = np.zeros([l, l])
    for i in range(l):
        col = (i + 1) % l
        result[i][col] = 1
    return result

def make_identity(l: int) -> np.array:
    result = np.zeros([l, l])
    for i in range(l):
        result[i][i] = 1
    return result

def make_y_matrix(l: int, m: int):
    s = make_cyclic_shift_matrix(m)
    identity = make_identity(l)
    return np.kron(identity, s)


def make_x_matrix(l: int, m: int):
    s = make_cyclic_shift_matrix(l)
    identity = make_identity(m)
    return np.kron(s, identity)

def make_xy_matrix(l: int, m: int):
    x = make_x_matrix(l, m)
    y = make_y_matrix(l, m)
    return np.matmul(x, y)

def create_hx_and_hz_parity_check(l: int, m: int, a: List[int], b: List[int]):
    A_x1 = np.linalg.matrix_power(make_x_matrix(l, m), a[0])
    A_y1 = np.linalg.matrix_power(make_y_matrix(l, m), a[1])
    A_y2 = np.linalg.matrix_power(make_y_matrix(l, m), a[2])
    A_matrix = (A_x1 + A_y1 + A_y2) % 2

    B_y1 = np.linalg.matrix_power(make_y_matrix(l, m), b[0])
    B_x1 = np.linalg.matrix_power(make_x_matrix(l, m), b[1])
    B_x2 = np.linalg.matrix_power(make_x_matrix(l, m), b[2])
    B_matrix = (B_y1 + B_x1 + B_x2) % 2

    Hx = np.hstack((A_matrix, B_matrix))
    Hz = np.hstack((np.transpose(B_matrix), np.transpose(A_matrix)))
    return Hx, Hz

def create_coprime_hx_and_hx(l: int, m: int, a: List[int], b: List[int]):
    A_x1 = np.linalg.matrix_power(make_xy_matrix(l, m), a[0])
    A_y1 = np.linalg.matrix_power(make_xy_matrix(l, m), a[1])
    A_y2 = np.linalg.matrix_power(make_xy_matrix(l, m), a[2])
    A_matrix = (A_x1 + A_y1 + A_y2) % 2

    B_y1 = np.linalg.matrix_power(make_xy_matrix(l, m), b[0])
    B_x1 = np.linalg.matrix_power(make_xy_matrix(l, m), b[1])
    B_x2 = np.linalg.matrix_power(make_xy_matrix(l, m), b[2])
    B_matrix = (B_y1 + B_x1 + B_x2) % 2

    Hx = np.hstack((A_matrix, B_matrix))
    Hz = np.hstack((np.transpose(B_matrix), np.transpose(A_matrix)))
    return Hx, Hz

def create_coprime_parity_check(l: int, m: int, a: List[int], b: List[int]):
    hx, hz = create_coprime_hx_and_hx(l, m, a, b)
    top = np.hstack((hx, np.zeros([len(hx), len(hz[0])])))
    bottom = np.hstack((np.zeros([len(hz), len(hx[0])]), hz))
    H = np.vstack((top, bottom))
    return H

def create_full_parity_check(l: int, m: int, a: List[int], b: List[int]):
    hx, hz = create_hx_and_hz_parity_check(l, m, a, b)
    top = np.hstack((hx, np.zeros([len(hx), len(hz[0])])))
    bottom = np.hstack((np.zeros([len(hz), len(hx[0])]), hz))
    H = np.vstack((top, bottom))
    return H

