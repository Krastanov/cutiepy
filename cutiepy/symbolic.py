'''
Computer Algebra Core
=====================

This module permits the creation of symbolic representation of mathematical
operations.

Implementation details
======================

The code here is meant to be simple at the expense of not being easy to
generalize or extend. Repetition of code is preferable over a complicated class
structure. Slow but simple code is preferable over fast code. This is not
supposed to be an implementation of a competitive computer algebra system,
rather the bare minimum necessary for quantum mechanics. It is used for code
generation more than it is used for analytical calculations.

Canonicalization is done in the ``__new__`` methods.

Checks for the canonicalization contracts are done in the ``__init__`` methods
and are based on the ``_postcanon`` methods.

Equality checks and hashing are inherited from ``str`` for ``Atom`` and from
``tuple`` for ``Node``, so we do not have to worry about them. Hence two atoms
with the same name will evaluate to ``True`` in the ``==`` operator. Never
create two atoms with the same name.

``shapetype`` and ``dims`` are calculated recursively. ``shape`` is calculated
from the two of them.

Only square operators and superoperators are supported for the moment.

No unevaluated symbolic derivatives. No fancy numeric types, just the python
``numbers`` module.

Printing to the console is not prettified. Printing in the notebook is done
with latex. ``_repr_latex_`` calls the recursive multimethod ``latex`` in the
general case. If additional information needs to be printed ``_repr_latex_``
can be overwritten as it is never called recursively. ``latex`` uses the
private method ``_latex``.

Classes starting with `_CG_` are only for the code generator, not for human
use.
'''

import functools
import itertools
import numbers
import operator
import uuid

import numpy as np

##############################################################################
# Expression tree.
##############################################################################
class Expr(object):
    '''Base class for all symbolic expressions.'''
    def __init__(self, *args):
        for cls in type(self).mro():
            if hasattr(cls, '_postcanon'):
                cls._postcanon(self)

    def __add__(self, other):
        return Add(self, other)
    def __sub__(self, other):
        return Add(self, Mul(-1, other))
    def __pow__(self, other):
        return Pow(self, other)
    def __radd__(self, other):
        return Add(other, self)
    def __rsub__(self, other):
        return Add(other, Mul(-1, self))
    def __rpow__(self, other):
        return Pow(other, self)
    def __neg__(self):
        return Mul(-1, self)
    def __truediv__(self, other):
        return Mul(self, Pow(other, -1))
    def __rtruediv__(self, other):
        return Mul(other, Pow(self, -1))
    def __mul__(self, other):
        scalars, not_scalars = split_by_predicate(isscalar, [self, other])
        return Mul(Mul(*scalars), Dot(*not_scalars))
    def __rmul__(self, other):
        return Expr.__mul__(other, self)

    def _repr_latex_(self):
        return '$$%s$$'%latex(self)


##############################################################################
# Expression tree - Atomic objects.
##############################################################################
class Atom(Expr, str):
    '''Base class for atomic expressions.'''
    _anon = False
    @classmethod
    def anon(cls, *args, **kwargs):
        '''Create an object with unique name.'''
        res = cls('%s'%uuid.uuid4(),
                  *args, **kwargs)
        res._anon = True
        return res


class Scalar(Atom):
    '''Represent real scalar variables.'''
    def _latex(self):
        if self._anon:
            return r'\tiny\boxed{{%s}_{%s}}\normalsize'%(shapetype(self).__name__[0],
                                                         self[:8]+'...')
        return str(self)


class NotScalar(Atom):
    '''Base class for non-scalar objects.'''
    def __new__(cls, name, dims, numerical=None):
        self = Atom.__new__(cls, name)
        if isinstance(dims, numbers.Integral):
            self._ds = [dims]
        else:
            self._ds = dims
        self.numerical = numerical
        if numerical is not None:
            eshape = shape(self)
            if eshape != numerical.shape:
                numerical.shape = eshape
        return self

    def _dims(self):
        return self._ds

    def _base_sub(self):
        if self._anon:
            return (r'\tiny\boxed{{%s}_{%s}}\normalsize'%(shapetype(self).__name__[0],
                                                          str(self)[:8]+'...'),
                    '')
        return tuple((self.split('_')+[''])[:2])

    def _repr_latex_(self):
        if self._anon:
            s = r'\text{Anonymous }'
        else:
            s = ''
        s += r'\text{%s }%s \text{ on the space }'%(shapetype(self).__name__, latex(self))
        s += r'\otimes'.join(r'\mathbb{C}^{%d}'%_ for _ in self._ds)
        if self.numerical is None:
            s += r'\text{ without an attached numerical content.}'
        else:
            s += r'\text{ with numerical content: }'+'$$\n$$'
            s += numerical_matrix_latex(self.numerical)
        return '$$%s$$'%s


class Ket(NotScalar):
    @staticmethod
    def _dim_to_shape(dim):
        '''
        >>> shape(Ket('v',10))
        (10, 1)
        '''
        return (dim, 1)

    def _latex(self):
        return r'| {%s}_{%s} \rangle'%self._base_sub()


class Bra(NotScalar):
    @staticmethod
    def _dim_to_shape(dim):
        '''
        >>> shape(Bra('c',10))
        (1, 10)
        '''
        return (1, dim)

    def _latex(self):
        return r'\langle {%s}_{%s} |'%self._base_sub()


class Operator(NotScalar):
    @staticmethod
    def _dim_to_shape(dim):
        '''
        >>> shape(Operator('O',10))
        (10, 10)
        '''
        return (dim, dim)

    def _latex(self):
        return r'\hat{%s}_{%s}'%self._base_sub()


class Superoperator(NotScalar):
    @staticmethod
    def _dim_to_shape(dim):
        '''
        >>> shape(Superoperator('S',10))
        (10, 10, 10, 10)
        '''
        return (dim, dim, dim, dim)

    def _latex(self):
        return r'\mathcal{{%s}_{%s}}'%self._base_sub()


##############################################################################
# Expression tree - Nodes.
##############################################################################
class Node(Expr, tuple):
    '''Base class for non-atomic expressions.'''
    def __new__(cls, *args):
        return super(Node, cls).__new__(cls, tuple(args))
    def __repr__(self):
        return type(self).__name__ + '(%s)'%', '.join(map(repr, self))
    def __str__(self):
        return self.__repr__()
    def _shape(self):
        shapetype(self)._dim_to_shape(sum(dims(self)))


class _CG_Node(Node):
    pass


class Add(Node):
    def __new__(cls, *expr):
        '''
        >>> x, y, z = xyz()
        >>> x+0 # Remove zeros.
        'x'
        >>> 1+x+1 # Numbers gathered and in front.
        Add(2, 'x')
        >>> x+2*x # `Mul`s gathered.
        Mul(3, 'x')
        '''
        # Flatten.
        flat = list(itertools.chain.from_iterable(_ if isinstance(_, Add) else (_,)
                                                  for _ in expr))
        # Sum numbers.
        nums, not_nums = split_by_predicate(isnumber, flat)
        nums_sum = sum(nums)
        # Gather monomials.
        monomial_tuples = [(_[0], _[1:]) if isinstance(_, Mul) and isnumber(_[0])
                           else (1, (_,))
                           for _ in not_nums]
        monomial_tuples.sort(key=lambda _: hash(_[1]))
        groups = itertools.groupby(monomial_tuples, lambda _: _[1])
        reduced = []
        for key, factors in groups:
            tot = sum(_[0] for _ in factors)
            if tot == 1:
                reduced.append(key[0] if len(key) == 1 else Mul(*key))
            elif tot != 0:
                reduced.append(Mul(tot, *key))
        # Sort and return.
        reduced.sort(key=hash)
        if nums_sum:
            reduced.insert(0, nums_sum)
        if len(reduced) == 0:
            return 0
        elif len(reduced) == 1:
            return reduced[0]
        else:
            return super(Add, cls).__new__(cls, *reduced)

    def _postcanon(self):
        '''
        >>> Ket('v1', 10)+Bra('c2', 10)
        Traceback (most recent call last):
            ...
        AssertionError: The shapes of the elements of Add(...) are not all the same.
        '''
        shape_0 = shape(self[0])
        assert all(shape(_) == shape_0 for _ in self[1:]),\
               'The shapes of the elements of %s are not all the same.'%(self,)

    def _shapetype(self):
        '''
        >>> shape(Scalar('x')+Scalar('y'))
        ()
        >>> shape(Ket('v1', 10)+Ket('v2', 10))
        (10, 1)
        '''
        return shapetype(self[0])

    def _dims(self):
        '''
        >>> dims(Scalar('x')+Scalar('y'))
        []
        >>> dims(Ket('v1', 10)+Ket('v2', 10))
        [10]
        '''
        return dims(self[0])

    def _latex(self):
        return r'\left( %s \right)'%'+'.join(map(latex, self))


class Mul(Node):
    def __new__(cls, *expr):
        '''
        >>> x, y, z = xyz()
        >>> x*0 # Remove zeros.
        0
        >>> x*1 # Remove ones.
        'x'
        >>> 2*x*2 # Numbers gathered and in front.
        Mul(4, 'x')
        >>> x*x**2 # `Pow`s gathered.
        Pow('x', 3)
        >>> Ket('v', 5)*x*3 # Numbers before scalars before the rest.
        Mul(3, 'x', 'v')
        '''
        # Flatten.
        flat = list(itertools.chain.from_iterable(_ if isinstance(_, Mul) else (_,)
                                                  for _ in expr))
        # Multiply numbers.
        nums, not_nums = split_by_predicate(isnumber, flat)
        nums_prod = functools.reduce(operator.mul, nums, 1)
        if nums_prod == 0:
            return 0
        # Gather monomials.
        monomial_tuples = [(_[:1], _[1]) if isinstance(_, Pow) and isnumber(_[1])
                           else ((_,), 1)
                           for _ in not_nums]
        monomial_tuples.sort(key=lambda _: hash(_[0]))
        groups = itertools.groupby(monomial_tuples, lambda _: _[0])
        reduced = []
        for key, powers in groups:
            tot = sum(_[1] for _ in powers)
            if tot == 1:
                reduced.append(key[0])
            elif tot != 0:
                reduced.append(Pow(key[0], tot))
        # Sort and return.
        reduced.sort(key=hash)
        scalars, not_scalars = split_by_predicate(isscalar, reduced)
        reduced = list(scalars)+list(not_scalars)
        if nums_prod != 1:
            reduced.insert(0, nums_prod)
        if len(reduced) == 0:
            return 1
        elif len(reduced) == 1:
            return reduced[0]
        else:
            return super(Mul, cls).__new__(cls, *reduced)

    def _postcanon(self):
        '''
        >>> Node.__new__(Mul, Ket('v1', 10), Ket('v2', 10)).__init__()
        Traceback (most recent call last):
            ...
        AssertionError: More than one of the elements of Mul(...) are not scalars.
        >>> Node.__new__(Mul, Ket('v1', 10), Scalar('x')).__init__()
        Traceback (most recent call last):
            ...
        AssertionError: The non scalar in Mul(...) is not at last position.
        '''
        non_scalars = sum(not isscalar(_) for _ in self)
        if non_scalars:
            assert non_scalars == 1, 'More than one of the elements of %s are not scalars.'%(self,)
            assert not isscalar(self[-1]), 'The non scalar in %s is not at last position.'%(self,)

    def _shapetype(self):
        '''
        >>> shape(Scalar('x')*Scalar('y'))
        ()
        >>> shape(Scalar('x')*Ket('v2', 10))
        (10, 1)
        '''
        return shapetype(self[-1])

    def _dims(self):
        '''
        >>> dims(Scalar('x')*Scalar('y'))
        []
        >>> dims(Scalar('x')*Ket('v2', 10))
        [10]
        '''
        return dims(self[-1])

    def _latex(self):
        return r'\tiny\times\normalsize'.join(map(latex, self))


class Pow(Node):
    def __new__(cls, base, exponent):
        '''
        >>> x, y, z = xyz()
        >>> x**0 # Remove zero powers.
        1
        >>> Pow(0,2) # Remove zero bases.
        0
        >>> Pow(0,0) # Most useful definition.
        1
        >>> x**1 # Remove one powers.
        'x'
        >>> 1**x # Remove one bases.
        1
        '''
        if base == 1:
            return 1
        elif base == 0:
            if exponent == 0:
                return 1
            elif isnumber(exponent) and exponent > 0:
                return 0
            return super(Pow, cls).__new__(cls, base, exponent)
        elif exponent == 1:
            return base
        elif exponent == 0:
            return 1
        return super(Pow, cls).__new__(cls, base, exponent)

    def _postcanon(self):
        '''
        >>> shape(Scalar('x')**Ket('v2', 10))
        Traceback (most recent call last):
            ...
        AssertionError: The exponent in Pow(...) is not a scalar.
        >>> shape(Ket('v1', 10)**Scalar('x'))
        Traceback (most recent call last):
            ...
        AssertionError: The base in Pow(...) is not square.
        '''
        assert isscalar(self[1]), 'The exponent in %s is not a scalar.'%(self,)
        assert isscalar(self[0]) or isoperator(self[0]) or issuperoperator(self[0]),\
               'The base in %s is not square.'%(self,)

    def _shapetype(self):
        '''
        >>> shape(Scalar('x')**Scalar('y'))
        ()
        >>> shape(Operator('O', 10)**Scalar('x'))
        (10, 10)
        '''
        return shapetype(self[0])

    def _dims(self):
        '''
        >>> dims(Scalar('x')**Scalar('y'))
        []
        >>> dims(Operator('O', 10)**Scalar('x'))
        [10]
        '''
        return dims(self[0])

    def _latex(self):
        return r'{ \left( %s \right) }^{%s}'%(latex(self[0]), latex(self[1]))


class Dot(Node):
    def __new__(cls, *expr):
        '''
        >>> x, y, z = xyz()
        >>> v, c, O, S = Ket('v',10), Bra('c',10), Operator('O', 10), Superoperator('S', 10)
        >>> c*S*S*O**2*O*v # `Pow`s gathered.
        Dot('c', Pow('S', 2), Pow('O', 3), 'v')
        >>> S*(x*O) # Take `Mul`s out of `Dot`s
        Mul('x', Dot('S', 'O'))
        '''
        # Flatten.
        flat = list(itertools.chain.from_iterable(_ if isinstance(_, Dot) else (_,)
                                                  for _ in expr))
        # Gather monomials.
        monomial_tuples = [(_[:1], _[1]) if isinstance(_, Pow) and isnumber(_[1])
                           else ((_,), 1)
                           for _ in flat]
        groups = itertools.groupby(monomial_tuples, lambda _: _[0])
        reduced = []
        for key, powers in groups:
            tot = sum(_[1] for _ in powers)
            if tot == 1:
                reduced.append(key[0])
            elif tot != 0:
                reduced.append(Pow(key[0], tot))
        # Return.
        if len(reduced) == 0:
            return 1
        elif len(reduced) == 1:
            return reduced[0]
        else:
            # Take `Mul`s out.
            mul = []
            dot = []
            for el in reduced:
                if isinstance(el, Mul):
                    mul.extend(el[0:-1])
                    dot.append(el[-1])
                else:
                    dot.append(el)
            if mul:
                mul.append(Dot(*dot))
                return Mul(*mul)
            return super(Dot, cls).__new__(cls, *reduced)

    def _postcanon(self):
        '''
        >>> v, c, O, S = Ket('v',10), Bra('c',10), Operator('O', 10), Superoperator('S', 10)
        >>> c*Ket('v1', 11)
        Traceback (most recent call last):
            ...
        AssertionError: Not all elements of Dot(...) have the same dimensions.
        >>> v*O
        Traceback (most recent call last):
            ...
        AssertionError: The elements of Dot(...) are not in covec, supop, op, vec order.
        >>> v*Ket('v2', 10)
        Traceback (most recent call last):
            ...
        AssertionError: Dot(...) contains more than one (co)ket.
        '''
        d = dim(self[0])
        assert all(dim(_) == d for _ in self[1:]),\
               'Not all elements of %s have the same dimensions.'%(self,)

        order = {0:isbra, 1:issuperoperator, 2:isoperator, 3:isket}
        current = 0
        for elem in self:
            while not order[current](elem):
                current += 1
                assert current < 4,\
                       'The elements of %s are not in covec, supop, op, vec order.'%(self,)

        assert sum(map(isket, self)) < 2 and sum(map(isbra, self)) < 2,\
               '%s contains more than one (co)ket.'%(self,)

    def _shapetype(self):
        '''
        >>> v, c, O, S = Ket('v',10), Bra('c',10), Operator('O', 10), Superoperator('S', 10)
        >>> shape(c*v)
        ()
        >>> shape(S*O)
        (10, 10)
        >>> shape(S*S)
        (10, 10, 10, 10)
        >>> shape(c*O)
        (1, 10)
        '''
        if isket(self[-1]):
            if isbra(self[0]):
                return Scalar
            return shapetype(self[-1])
        if isbra(self[0]):
            return shapetype(self[0])
        if issuperoperator(self[0]):
            if isoperator(self[-1]):
                return shapetype(self[-1])
        return shapetype(self[0])

    def _dims(self):
        '''
        >>> v, c, O, S = Ket('v',10), Bra('c',10), Operator('O', 10), Superoperator('S', 10)
        >>> dims(c*v)
        []
        >>> dims(S*O)
        [10]
        >>> dims(S*S)
        [10]
        >>> dims(c*O)
        [10]
        '''
        if isscalar(self):
            return []
        return dims(self[0])

    def _latex(self):
        return ''.join(map(latex, self))


class TensorProd(Node):
    def __new__(cls, *expr):
        '''
        >>> O = Operator('O', 10)
        >>> TensorProd(O,TensorProd(O,O)) # Flatten.
        TensorProd('O', 'O', 'O')
        '''
        # Flatten.
        flat = list(itertools.chain.from_iterable(_ if isinstance(_, TensorProd) else (_,)
                                                  for _ in expr))
        if len(flat) == 1:
            return flat[0]
        return super(TensorProd, cls).__new__(cls, *flat)

    def _postcanon(self):
        '''
        >>> v, c = Ket('v',10), Bra('c',10)
        >>> TensorProd(2, v)
        Traceback (most recent call last):
            ...
        AssertionError: TensorProd(...) contains scalars.
        >>> TensorProd(c, v)
        Traceback (most recent call last):
            ...
        AssertionError: TensorProd(...) contains more than one type of elements ...
        '''
        shapetypes = set(map(shapetype, self))
        assert Scalar not in shapetypes, '%s contains scalars.'%(self,)
        assert len(shapetypes) == 1,\
               '%s contains more than one type of elements (kets, operators, etc).'%(self,)

    def _shapetype(self):
        '''
        >>> v1, v2 = Ket('v1', 2), Ket('v2', 5)
        >>> isket(TensorProd(v1, v2))
        True
        '''
        return shapetype(self[0])

    def _dims(self):
        '''
        >>> v1, v2 = Ket('v1', 2), Ket('v2', 5)
        >>> dims(TensorProd(v1, v2))
        [2, 5]
        '''
        return list(itertools.chain.from_iterable(map(dims, self)))

    def _latex(self):
        if shapetype(self) is Ket:
            return r'|%s\rangle'%','.join(self)
        if shapetype(self) is Bra:
            return r'\langle %s|'%','.join(self)
        return r'\tiny\otimes\normalsize'.join(map(latex, self))
tensor = TensorProd


class Commutator(Node):
    def __new__(cls, A, B):
        if A == B:
            return 0
        return super(Commutator, cls).__new__(cls, A, B)

    def _postcanon(self):
        assert shapetype(self[0]) == shapetype(self[1]) == Operator, '%s should contain only Operators.'%(self,)
        assert dims(self[0]) == dims(self[1]), '%s contains operators of incompatible dimmensions.'%(self,)

    def _shapetype(self):
        return shapetype(self[0])

    def _dims(self):
        return dims(self[0])

    def _latex(self):
        return r'\left[ {%s}, {%s} \right]'%(latex(self[0]), latex(self[1]))


class _CG_AppliedLindbladSuperoperator(_CG_Node):
    def __new__(cls, C, rho):
        if C == 0 or rho == 0:
            return 0
        return super(_CG_AppliedLindbladSuperoperator, cls).__new__(cls, C, rho)

    def _postcanon(self):
        assert shapetype(self[0]) == Operator, '%s should contain an Operator as a collapse operator.'%(self,)
        assert shapetype(self[1]) == Operator, '%s should contain an Operator as a density matrix.'%(self,)
        assert dims(self[0]) == dims(self[1]), 'The collapse operator and the density matrix are not of the same dimensions in %s.'%(self,)

    def _shapetype(self):
        return Operator

    def _dims(self):
        return dims(self[0])


class Trace(Node):
    pass


class Exp(Node):
    pass


class Log(Node):
    pass


class Dag(Node):
    pass


##############################################################################
# Expression tree - Functions on scalars.
##############################################################################
class ScalarFunction(Node):
    def _postcanon(self):
        '''
        >>> v = Ket('v',10)
        >>> ScalarFunction(v)
        Traceback (most recent call last):
            ...
        AssertionError: The arguments of ScalarFunction(...) are not scalars.
        '''
        assert shapetype(self[0]) is Scalar, 'The arguments of %s are not scalars.'%(self,)

    def _shapetype(self):
        return Scalar

    def _latex(self):
        return r'\operatorname{%s}\left( {%s} \right)'%(type(self).__name__, latex(self[0]))


class sin(ScalarFunction):
    pass

class cos(ScalarFunction):
    pass

class exp(ScalarFunction):
    pass

class sinh(ScalarFunction):
    pass

class cosh(ScalarFunction):
    pass

class tan(ScalarFunction):
    pass

class tanh(ScalarFunction):
    pass

class log(ScalarFunction):
    pass


##############################################################################
# Multimethods
##############################################################################
# Using multimethods instead of class methods because they have to work on
# numbers as well (and numbers are not custom subclasses).
def shapetype(expr):
    if isnumber(expr):
        return Scalar
    if isinstance(expr, (Scalar, Ket, Bra, Operator, Superoperator)):
        return type(expr)
    return expr._shapetype()

def dims(expr):
    if isscalar(expr):
        return []
    return expr._dims()

def isscalar(expr):
    return shapetype(expr) is Scalar

def isket(expr):
    return shapetype(expr) is Ket

def isbra(expr):
    return shapetype(expr) is Bra

def isoperator(expr):
    return shapetype(expr) is Operator

def issuperoperator(expr):
    return shapetype(expr) is Superoperator

def dim(expr):
    return functools.reduce(operator.mul, dims(expr))

def shape(expr):
    if isscalar(expr):
        return ()
    return shapetype(expr)._dim_to_shape(dim(expr))

def size(expr):
    return functools.reduce(operator.mul, shape(expr))

def latex(expr):
    if isnumber(expr):
        return '{%s}'%str(expr)
    return '{%s}'%expr._latex()

def subs(expr, key, value):
    if expr == key:
        return value
    if isinstance(expr, (numbers.Number, Atom)):
        return expr
    return type(expr)(*(value if key == _ else subs(_, key, value)
                        for _ in expr))

def isnumerical(expr):
    if isnumber(expr):
        return True
    elif isscalar(expr) or isinstance(expr, Node):
        return False
    return expr.numerical is not None

def numerical(expr):
    if isnumber(expr):
        return expr
    return expr.numerical


##############################################################################
# Helpers.
##############################################################################
def split_by_predicate(p, l):
    '''Separate an iterable in two iterators depending on predicate.'''
    t1, t2 = itertools.tee(l)
    return filter(p, t1), itertools.filterfalse(p, t2)


def isnumber(expr):
    '''Check based on Abstract Base Classes.'''
    return isinstance(expr, numbers.Number)


def numerical_matrix_latex(matrix):
    # pylint: disable=too-many-branches,unused-argument,no-member,unidiomatic-typecheck,invalid-name,undefined-loop-variable
    # Copied from qutip
    M, N = matrix.shape
    s = r'\begin{equation*}\left(\begin{array}{*{11}c}'
    def _format_float(value):
        if value == 0.0:
            return '0.0'
        elif abs(value) > 1000.0 or abs(value) < 0.001:
            return ('%.3e' % value).replace('e', r'\times10^{') + '}'
        elif abs(value - int(value)) < 0.001:
            return '%.1f' % value
        else:
            return '%.3f' % value
    def _format_element(m, n, d):
        s = ' & ' if n > 0 else ''
        if type(d) == str:
            return s + d
        else:
            if abs(np.imag(d)) < 1e-12:
                return s + _format_float(np.real(d))
            elif abs(np.real(d)) < 1e-12:
                return s + _format_float(np.imag(d)) + 'j'
            else:
                s_re = _format_float(np.real(d))
                s_im = _format_float(np.imag(d))
                if np.imag(d) > 0.0:
                    return s + '(' + s_re + '+' + s_im + 'j)'
                else:
                    return s + '(' + s_re + s_im + 'j)'
    if M > 10 and N > 10:
        # truncated matrix output
        for m in range(5):
            for n in range(5):
                s += _format_element(m, n, matrix[m, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(m, n, matrix[m, n])
            s += r'\\'
        for n in range(5):
            s += _format_element(m, n, r'\vdots')
        s += r' & \ddots'
        for n in range(N - 5, N):
            s += _format_element(m, n, r'\vdots')
        s += r'\\'
        for m in range(M - 5, M):
            for n in range(5):
                s += _format_element(m, n, matrix[m, n])
            s += r' & \cdots'
            for n in range(N - 5, N):
                s += _format_element(m, n, matrix[m, n])
            s += r'\\'
    elif M > 10 and N == 1:
        # truncated column ket output
        for m in range(5):
            s += _format_element(m, 0, matrix[m, 0])
            s += r'\\'
        s += _format_element(m, 0, r'\vdots')
        s += r'\\'
        for m in range(M - 5, M):
            s += _format_element(m, 0, matrix[m, 0])
            s += r'\\'
    elif M == 1 and N > 10:
        # truncated row ket output
        for n in range(5):
            s += _format_element(0, n, matrix[0, n])
        s += r' & \cdots'
        for n in range(N - 5, N):
            s += _format_element(0, n, matrix[0, n])
        s += r'\\'
    else:
        # full output
        for m in range(M):
            for n in range(N):
                s += _format_element(m, n, matrix[m, n])
            s += r'\\'
    s += r'\end{array}\right)\end{equation*}'
    return s


def xyz():
    return [Scalar(_) for _ in 'xyz']

t = Scalar('t')
