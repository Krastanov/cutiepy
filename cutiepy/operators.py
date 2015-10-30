import numpy as np
from scipy.sparse import csr_matrix
from .symbolic import Operator

SPARSITY_N_CUTOFF = 600 # TODO lower after fixing sparse matrices

def sparsify(mat):
    assert SPARSITY_N_CUTOFF > 5, 'The SPARSITY_N_CUTOFF is set to a very low number.'
    if min(mat.shape) > SPARSITY_N_CUTOFF:
        return csr_matrix(mat)
    return mat


def destroy(N):
    return Operator('{a}_{%d}'%N,
                    N,
                    sparsify(np.diag(np.arange(1,N,dtype=complex)**0.5,k=1)))

def create(N): #TODO name prints ugly
    return Operator('{a^\dagger}_{%d}'%N,
                    N,
                    sparsify(np.diag(np.arange(1,N,dtype=complex)**0.5,k=-1)))

def num(N):
    return Operator('{n}_{%d}'%N,
                    N,
                    sparsify(np.diag(np.arange(0,N,dtype=complex))))

def identity(N):
    return Operator('{I}_{%d}'%N,
                    N,
                    sparsify(np.eye(N,dtype=complex)))

def randomH(N):
    m = np.random.random([N,N]) + 1j*np.random.random([N,N])
    m = (m + np.conj(m.T))/2
    return Operator.anon(N, sparsify(m))

s_m = np.array([[0, 0 ],[1 , 0]],dtype=complex)
s_p = np.array([[0, 1 ],[0 , 0]],dtype=complex)
s_x = np.array([[0, 1 ],[1 , 0]],dtype=complex)
s_y = np.array([[0,-1j],[1j, 0]],dtype=complex)
s_z = np.array([[1, 0 ],[0 ,-1]],dtype=complex)
def sigmam():
    return Operator('σ_-', 2, s_m)
def sigmap():
    return Operator('σ_+', 2, s_p)
def sigmax():
    return Operator('σ_x', 2, s_x)
def sigmay():
    return Operator('σ_y', 2, s_y)
def sigmaz():
    return Operator('σ_z', 2, s_z)
