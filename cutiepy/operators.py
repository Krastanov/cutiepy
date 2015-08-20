import numpy as np

from .symbolic import Operator

def destroy(N):
    return Operator('{a}_{%d}'%N,N,np.diag(np.arange(1,N,dtype=complex)**0.5,k=1))
def create(N): #TODO name prints ugly
    return Operator('{a^\dagger}_{%d}'%N,N,np.diag(np.arange(1,N,dtype=complex)**0.5,k=-1))
def num(N):
    return Operator('{n}_{%d}'%N,N,np.diag(np.arange(0,N,dtype=complex)))
def identity(N):
    return Operator('{I}_{%d}'%N,N,np.eye(N,dtype=complex))

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
