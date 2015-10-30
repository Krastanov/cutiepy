from .evalf import evalf
from .symbolic import numerical, isket

import numpy as np
import scipy

def sparse_aware_dot(l,r): # XXX This should be in scipy!?
    if scipy.sparse.issparse(l):
        return l.dot(r)
    elif scipy.sparse.issparse(r):
        return type(r).dot(l,r)
    return l.dot(r)
dot = sparse_aware_dot

def expect(operators, states, keep_complex=False):
    if not isinstance(states, list):
        states = [states]
    if not isinstance(operators, list):
        operators = [operators]
    nstates = [numerical(evalf(_)) for _ in states]
    noperators = [numerical(evalf(_)) for _ in operators]
    if isket(states[0]):
        ret = [[dot(s.T.conj(), dot(o, s))[0,0] for s in nstates] for o in noperators]
    else:
        ret = [[np.trace(dot(o, s)) for s in nstates] for o in noperators]
    ret = np.array(ret, dtype=np.complex).T
    if not keep_complex:
        ret = np.real(ret)
    return ret

def overlap(states_a, states_b):
    if not isinstance(states_a, list):
        states_a = [states_a]
    if not isinstance(states_b, list):
        states_b = [states_b]
    states_a = [numerical(evalf(_)) for _ in states_a]
    states_b = [numerical(evalf(_)) for _ in states_b]
    ret = [[dot(b.T.conj(), a)[0,0] for b in states_b] for a in states_a]
    return np.array(ret, dtype=np.complex).T
