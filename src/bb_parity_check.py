from typing import Callable, List, Tuple, Union
import numpy as np

def make_cyclic_shift_matrix(l: int) -> np.ndarray:
    """Cyclic shift matrix of size l x l."""
    return np.roll(np.eye(l), shift=1, axis=1)

def make_y_matrix(l: int, m: int) -> np.ndarray:
    """Kronecker product for Y matrix."""
    return np.kron(np.eye(l), make_cyclic_shift_matrix(m))

def make_x_matrix(l: int, m: int) -> np.ndarray:
    """Kronecker product for X matrix."""
    return np.kron(make_cyclic_shift_matrix(l), np.eye(m))

def make_xy_matrix(l: int, m: int, px: int, py: int) -> np.ndarray:
    """Matrix for x^px y^py (note: x and y may not commute in general)."""
    X = np.linalg.matrix_power(make_x_matrix(l, m), px)
    Y = np.linalg.matrix_power(make_y_matrix(l, m), py)
    return (X @ Y) % 2

PolyTerm = Tuple[str, Union[int, Tuple[int, int]]]

def term_to_matrix(l: int, m: int, term: PolyTerm) -> np.ndarray:
    """Convert a polynomial term into its matrix representation."""
    name, power = term
    if name == 'x':
        return np.linalg.matrix_power(make_x_matrix(l, m), power)
    elif name == 'y':
        return np.linalg.matrix_power(make_y_matrix(l, m), power)
    elif name == 'xy':
        px, py = power
        return make_xy_matrix(l, m, px, py)
    else:
        raise ValueError(f"Unknown polynomial term: {name}")

def poly_to_matrix(l: int, m: int, poly: List[PolyTerm]) -> np.ndarray:
    """Convert a polynomial (list of terms) to its matrix representation."""
    total = np.zeros((l*m, l*m), dtype=int)
    for term in poly:
        total = (total + term_to_matrix(l, m, term)) % 2
    return total

def create_hx_and_hz_parity_check(l: int, m: int,
                                  a: List[Tuple[str, int]],
                                  b: List[Tuple[str, int]]):
    A_matrix = poly_to_matrix(l, m, a)
    B_matrix = poly_to_matrix(l, m, b)
    Hx = np.hstack((A_matrix, B_matrix))
    Hz = np.hstack((B_matrix.T, A_matrix.T))
    return Hx, Hz


def create_full_parity_check(l: int, m: int,
                             a: List[Tuple[str, int]],
                             b: List[Tuple[str, int]]):
    hx, hz = create_hx_and_hz_parity_check(l, m, a, b)
    top = np.hstack((hx, np.zeros([len(hx), len(hz[0])])))
    bottom = np.hstack((np.zeros([len(hz), len(hx[0])]), hz))
    H = np.vstack((top, bottom))
    return H