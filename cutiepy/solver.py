import scipy
import numpy as np

from .evalf import evalf
from .symbolic import dims, Ket, t, numerical, dims
from .progressbar import DoNothingBar, ProgressBar
from .codegen import ODESolver, generate_cython

def sesolve(H, state0, tlist,
            progress=False,
            mxsteps=1000, rtol=1e-6, atol=1e-8):
    y_anon = Ket('_sesolve%s'%(dims(H),),dims(H))
    cf = generate_cython(evalf(-1j*H*y_anon),
                         func=ODESolver(),
                         argument_order = [t, y_anon])
    ccf = cf.compiled()
    state0_dim = dims(state0)
    res = ccf.pythonsolve(tlist, numerical(evalf(state0)), mxsteps, rtol, atol)
    return [Ket.anon(state0_dim,_) for _ in res]
