import functools
import itertools
import numbers
import operator

import numpy as np
import scipy
import scipy.linalg

from .symbolic import (Atom, Scalar, ScalarFunction,
        Add, Mul, Pow, Dot, TensorProd, shapetype, dims, split_by_predicate,
        isscalar, isnumber, isnumerical, numerical)


@functools.lru_cache()
@functools.singledispatch
def evalf(expr):
    raise NotImplementedError('The following object can not be evaluated numerically: %s.'%(expr,))

@evalf.register(numbers.Complex)
def _(expr):
    return expr

@evalf.register(Atom)
def _(expr):
    return expr

@evalf.register(Add)
def _(expr):
    args_evalf = map(evalf, expr)
    numericals, notnumericals = split_by_predicate(isnumerical, args_evalf)
    numericals = list(numericals)
    if len(numericals) == 1:
        return numericals[0] + sum(notnumericals)
    else:
        num = sum(numerical(_) for _ in numericals)
        if num is not 0:
            return shapetype(expr).anon(dims(expr), num)+sum(notnumericals)
        return sum(notnumericals)

@evalf.register(Mul)
def _(expr):
    prod = lambda _: functools.reduce(operator.mul, _, 1)
    args_evalf = map(evalf, expr)
    numericals, notnumericals = split_by_predicate(isnumerical, args_evalf)
    numericals = list(numericals)
    if len(numericals) == 1:
        return numericals[0] * prod(notnumericals)
    else:
        num = prod(numerical(_) for _ in numericals)
        if num is not 1:
            if isnumber(num):
                return num*prod(notnumericals)
            return shapetype(expr).anon(dims(expr), num)*prod(notnumericals)
        return sum(notnumericals)

@evalf.register(Pow)
def _(expr):
    b, e = expr
    if isscalar(b):
        return evalf(b)**evalf(e)
    elif isinstance(e, numbers.Integral):
        return functools.reduce(np.dot, [evalf(b)]*int(e))
    raise NotImplementedError('Can not raise a matrix to a non-integer power in %s.'%(expr,))

@evalf.register(Dot)
def _(expr):
    groups = itertools.groupby(map(evalf, expr), isnumerical)
    reduced = []
    for key, group in groups:
        group = list(group)
        if key:
            num = functools.reduce(np.dot, map(numerical, group))
            sym = Dot(*group)
            reduced.append(shapetype(sym).anon(dims(sym), num))
        else:
            reduced.extend(group)
    return Dot(*reduced)

kron = lambda l,r: scipy.sparse.kron(l,r, 'csr')
@evalf.register(TensorProd)
def _(expr):
    groups = itertools.groupby(map(evalf, expr), isnumerical)
    reduced = []
    for key, group in groups:
        group = list(group)
        if key:
            num = functools.reduce(kron, map(numerical, group))
            sym = TensorProd(*group)
            reduced.append(shapetype(sym).anon(dims(sym), num))
        else:
            reduced.extend(group)
    return TensorProd(*reduced)

@evalf.register(ScalarFunction)
def _(expr):
    arg = evalf(expr[0])
    if isnumber(arg):
        return np.__dict__[type(expr).__name__](arg)
    return type(expr)(arg)
