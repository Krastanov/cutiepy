import numpy as np
from scipy.sparse import csr_matrix

from .symbolic import Ket

# TODO optimize (and benchmark) dtypes
# TODO sparse matrices
# TODO density matrices


def basis(N, i):
    r = np.zeros((N, 1), dtype=complex)
    r[i] = 1
    return Ket(r'{%d}_{\tiny N\normalsize %d}'%(i,N), N, r)


def _sqrt_factorial(n_vec):
    # take the square root before multiplying
    return np.array([np.prod(np.sqrt(np.arange(1, n + 1))) for n in n_vec])


def coherent(N, alpha, offset=0, method='analytic'):
    if method == "operator" and offset == 0:

        x = basis(N, 0)
        a = destroy(N)
        D = (alpha * a.dag() - conj(alpha) * a).expm()
        return D * x

    elif method == "analytic" or offset > 0:
        n = np.arange(N) + offset
        data = np.exp(-abs(alpha)**2/2) * complex(alpha)**n / _sqrt_factorial(n)
        return Ket(r'{\tiny\alpha\normalsize %.2f}_{\tiny N\normalsize %d}'%(alpha,N), N, data)

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")
