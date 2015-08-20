from .evalf import evalf
from .symbolic import numerical

import numpy as np

def expect(operators, states, keep_complex=False):
    if not isinstance(states, list):
        states = [states]
    if not isinstance(operators, list):
        operators = [operators]
    states = [numerical(evalf(_)) for _ in states]
    operators = [numerical(evalf(_)) for _ in operators]

    ret = [[np.conj(s.T).dot(o).dot(s) for s in states] for o in operators]
    ret = np.array(ret, dtype=np.complex).squeeze(axis=(2,3)).T
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
    ret = [[np.conj(b.T).dot(a) for b in states_b] for a in states_a]
    return np.array(ret, dtype=np.complex).squeeze(axis=(2,3)).T
