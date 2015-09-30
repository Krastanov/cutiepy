import sys

import scipy
import numpy as np

from .evalf import evalf
from .symbolic import dims, Ket, t, numerical, dims
from .progressbar import ProgressBar
from .codegen import ODESolver, generate_cython

def sesolve(H, state0, tlist,
            mxsteps=1000, rtol=1e-6, atol=1e-8):
    y_anon = Ket('_sesolve%s'%(dims(H),),dims(H))
    print('Generating cython code...')
    sys.stdout.flush()
    cf = generate_cython(evalf(-1j*H*y_anon),
                         func=ODESolver(),
                         argument_order = [t, y_anon])
    print('Compiling cython code...')
    sys.stdout.flush()
    ccf = cf.compiled()
    state0_dim = dims(state0)
    state0_dense_array = numerical(evalf(state0))
    if not isinstance(state0_dense_array, np.ndarray):
        state0_dense_array = state0_dense_array.toarray()
    print('Running cython code...')
    sys.stdout.flush()
    res = ccf.pythonsolve(tlist, state0_dense_array, mxsteps, rtol, atol,
                          ProgressBar(len(tlist)))
    print('Formatting the output...')
    sys.stdout.flush()
    return [Ket.anon(state0_dim,_) for _ in res]
