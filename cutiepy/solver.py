import sys

import scipy
import numpy as np

from .evalf import evalf
from .symbolic import (dims, Ket, Operator, Commutator,
_CG_AppliedLindbladSuperoperator, t, numerical, dims, isket)
from .interactive import ProgressBar, iprint
from .codegen import ODESolver, generate_cython

def sesolve(H, state0, tlist,
            mxsteps=1000, rtol=1e-6, atol=1e-8):
    y_anon = Ket('_sesolve%s'%(dims(H),),dims(H))
    iprint('Generating cython code...')
    cf = generate_cython(evalf(-1j*H*y_anon),
                         func=ODESolver(),
                         argument_order = [t, y_anon])
    iprint('Compiling cython code...')
    ccf = cf.compiled()
    state0_dim = dims(state0)
    state0_dense_array = numerical(evalf(state0))
    if not isinstance(state0_dense_array, np.ndarray):
        state0_dense_array = state0_dense_array.toarray()
    iprint('Running cython code...')
    res = ccf.pythonsolve(tlist, state0_dense_array, mxsteps, rtol, atol,
                          ProgressBar(len(tlist)))
    iprint('Formatting the output...')
    return [Ket.anon(state0_dim,_) for _ in res]


def mesolve(H, c_ops, state0, tlist,
            mxsteps=1000, rtol=1e-6, atol=1e-8):
    y_anon = Operator('_mesolve%s'%(dims(H),),dims(H))
    iprint('Generating cython code...')
    f = evalf(-1j*Commutator(H,y_anon))+\
        sum(_CG_AppliedLindbladSuperoperator(evalf(_),y_anon) for _ in c_ops)
    cf = generate_cython(f,
                         func=ODESolver(),
                         argument_order = [t, y_anon])
    iprint('Compiling cython code...')
    ccf = cf.compiled()
    state0_dim = dims(state0)
    state0_dense_array = numerical(evalf(state0))
    if not isinstance(state0_dense_array, np.ndarray):
        state0_dense_array = state0_dense_array.toarray()
    if isket(state0):
        state0_dense_array = state0_dense_array.dot(np.conj(state0_dense_array.T))
    iprint('Running cython code...')
    res = ccf.pythonsolve(tlist, state0_dense_array, mxsteps, rtol, atol,
                          ProgressBar(len(tlist)))
    iprint('Formatting the output...')
    return [Operator.anon(state0_dim,_) for _ in res]
