'''
Code generation module
======================
'''

import collections
import functools
import numbers
import tempfile

import numpy as np

import cutiepy
from .symbolic import (Scalar, NotScalar, ScalarFunction, Add, Mul, Pow, Dot,
        TensorProd, dim, shape)


DEBUG = False


TypedName = collections.namedtuple('TypedName', ['type', 'name', 'allocate'])


def display_highlighted_source(source):
    '''For use inside an IPython notebook: Display highlighted source.'''
    from pygments import highlight
    from pygments.lexers import CythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.core.display import HTML, display
    display(HTML(highlight(source, CythonLexer(), HtmlFormatter(full=True))))


def uid():
    '''Return a generator of unique names.'''
    from itertools import chain, product, count
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWZYZ'
    return (''.join(_) for _ in chain.from_iterable(product(alphabet, repeat=c) for c in count(1)))


base_cython_function_template = '''
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
import numpy as np
from libc.math cimport sin, cos, exp, sinh, cosh, tan, tanh, log

# Declaration of global variables for
# - intermediate results
# - predefined constants
{globals_declarations}

# The generated cython code
cdef generated_function(
    # input arguments
{arguments_declaration},
    # output argument
    {return_value.type} {return_value.name}
    ):
    # indices and intermediate values for various matrix operations
    cdef int i, j, k, ii, jj, kk
    cdef double complex c
    # intermediate results
    {intermediate_results_globals}
    # predefined constants
    {predefined_constants_globals}
    # evaluating the function
{expressions}

# Function to deliver the constants from python to cython
cpdef setup_generated_function(
{setup_arguments_declarations}
    ):
    {predefined_constants_globals}
{setup_expressions}
'''
class BaseCythonFunction():
    '''Represents a single code file.'''
    main_call = ''
    def __init__(self):
        self.uid = uid()
        self.arguments = []      # list of TypedNames
        self.declarations = []   # list of TypedNames
        self.expressions = []    # string forms of expressions 'name = expression of names'
        self.return_value = None # TypedName
        self.memoized = {}
        self.setup_arguments = []
        self._compiled_mod = None
    def globals_declaratioins_s(self):
        s = '\n'.join('cdef {0.type} {0.name} = {0.allocate}'.format(_)
                      for _ in self.declarations[:-1]+self.setup_arguments)
        return s if s else '# no global declarations'
    def intermediate_results_globals_s(self):
        s = ', '.join(_.name for _ in self.declarations)
        return 'global %s'%s if s else '# no intermediate results globals'
    def predefined_constants_globals_s(self):
        s = ', '.join(_.name for _ in self.setup_arguments)
        return 'global %s'%s if s else '# no predefined constants globals'
    def arguments_s(self):
        s = ',\n'.join('    {0.type} {0.name}'.format(_) for _ in self.arguments)
        return s if s else '# no arguments'
    def expressions_s(self):
        s = '\n'.join('    %s'%_ for _ in self.expressions)
        return s if s else '    # no calculations'
    def setup_arguments_s(self):
        s = ',\n'.join('    {0.type} _{0.name}'.format(_) for _ in self.setup_arguments)
        return s if s else '# no arguments'
    def setup_expressions_s(self):
        s = '\n'.join('    {0.name} = _{0.name}'.format(_) for _ in self.setup_arguments)
        return s if s else '    pass # no setup'
    def argument_order(self, args):
        self.setup_arguments = list(set(self.arguments+self.setup_arguments)
                                    -{self.memoized[_] for _ in args if _ in self.memoized})
        def get_typedname(a):
            if a in self.memoized:
                pass
            elif isinstance(a, Scalar):
                self.memoized[a] = TypedName('double', next(self.uid), '0')
            elif isinstance(a, Ket):
                self.memoized[a] = TypedName('double complex [:, :]', next(self.uid), '0')
            else:
                raise NotImplementedError('Unused argument with unknown cython type: %s'%a)
            return self.memoized[a]
        self.arguments = [get_typedname(_) for _ in args]
    def rev_memoized(self):
        return dict(zip(self.memoized.values(), self.memoized.keys()))
    def numerical_setup_arguments(self):
        return [self.rev_memoized()[_].numerical for _ in self.setup_arguments]
    def __str__(self):
        return base_cython_function_template.format(
                   globals_declarations = self.globals_declaratioins_s(),
                   intermediate_results_globals = self.intermediate_results_globals_s(),
                   predefined_constants_globals = self.predefined_constants_globals_s(),
                   arguments_declaration = self.arguments_s(),
                   expressions = self.expressions_s(),
                   return_value = self.return_value,
                   setup_arguments_declarations = self.setup_arguments_s(),
                   setup_expressions = self.setup_expressions_s(),
                   )
    def compiled(self):
        if self._compiled_mod is None:
            import imp
            import os
            import os.path
            from distutils.extension import Extension
            from pyximport.pyxbuild import pyx_to_dll
            import cutiepy
            if DEBUG:
                display_highlighted_source(str(self))
            with tempfile.NamedTemporaryFile(mode='w',prefix='cutiepy_tmp_',suffix='.pyx',dir='.') as f:
                f.write(str(self))
                f.flush()
                folder, filename = os.path.split(f.name)
                extname, _ = os.path.splitext(filename)
                extension = Extension(extname,
                              sources=[filename],
                              include_dirs=[os.path.join(os.path.dirname(cutiepy.__file__), 'include')],
                              libraries=["sundials_cvode", "sundials_nvecserial"],
                              extra_compile_args=["-O3"]
                            )
                try:
                    module_path = pyx_to_dll(filename, ext=extension)
                except Exception as e:
                    raise e
                finally:
                    os.remove(os.path.join(folder, extname+'.c'))
            self._compiled_mod = imp.load_dynamic(filename.split('.')[0],module_path)
        self._compiled_mod.setup_generated_function(*self.numerical_setup_arguments())
        return self._compiled_mod



_expr_to_func_memoized = {}
def generate_cython(expr, func, argument_order=[]):
    '''Generate a cython function from the given expression.

    Memoization wrapper around ``_generate_cython``.

    Memoizes at two levels:
        - globally ``expr`` to ``func``
        - within each ``func`` it memoizes subexpressions for common
          subexpression elimination.'''
    if argument_order and (expr, type(func), tuple(argument_order)) in _expr_to_func_memoized:
        return _expr_to_func_memoized[(expr, type(func), tuple(argument_order))]
    elif expr in func.memoized:
        func.return_value = func.memoized[expr]
    else:
        func = _generate_cython(expr, func)
        func.memoized[expr] = func.return_value
    if argument_order:
        func.argument_order(argument_order)
    _expr_to_func_memoized[(expr, type(func), tuple(argument_order))] = func
    return func


@functools.singledispatch
def _generate_cython(expr, func):
    raise NotImplementedError('The following object can not be translated to cython: %s.'%(expr,))

@_generate_cython.register(numbers.Complex)
def _(expr, func):
    func.return_value = TypedName('double complex', str(expr), '0')
    return func

@_generate_cython.register(numbers.Real)
def _(expr, func):
    func.return_value = TypedName('double', str(expr), '0')
    return func

@_generate_cython.register(Scalar)
def _(expr, func):
    uid = TypedName('double', next(func.uid), '0') # TODO using something better than next(func.uid) for more meaningful names
    func.arguments.append(uid)
    func.return_value = uid
    return func

@_generate_cython.register(NotScalar)
def _(expr, func):
    uid = TypedName('double complex [:, :]', next(func.uid), 'np.zeros(%s, dtype=np.complex)'%(shape(expr),))
    func.arguments.append(uid)
    func.return_value = uid
    return func

def variadic_add_m(args, out):
    return '''# {out} = {sum_comment}
    ii = {a}.shape[0]
    jj = {a}.shape[1]
    for i in range(ii):
        for j in range(jj):
            {out}[i,j] = {sum}'''.format(
    a = args[0].name,
    out = out.name,
    sum = '+'.join('%s[i,j]'%_.name for _ in args),
    sum_comment = '+'.join(_.name for _ in args))
@_generate_cython.register(Add)
def _(expr, func):
    ret_values = [generate_cython(_, func).return_value for _ in expr]
    uid = next(func.uid)
    s = shape(expr)
    alloc = 'np.zeros(%s, dtype=np.complex)'%(s,) if s else '0'
    ret = TypedName(ret_values[-1].type, uid, alloc)
    func.declarations.append(ret)
    if s:
        func.expressions.append(variadic_add_m(ret_values, ret))
    else:
        func.expressions.append("%s = %s"%(uid, '+'.join(_.name for _ in ret_values)))
    func.return_value = ret
    return func

def mul_s_m(s, m, out):
    return '''# {out} = {s}*{m}
    ii = {m}.shape[0]
    jj = {m}.shape[1]
    for i in range(ii):
        for j in range(jj):
            {out}[i,j] = {s}*{m}[i,j]'''.format(
    out = out,
    s = s,
    m = m)
@_generate_cython.register(Mul)
def _(expr, func):
    ret_values = [generate_cython(_, func).return_value for _ in expr]
    uid = next(func.uid)
    s = shape(expr)
    alloc = 'np.zeros(%s, dtype=np.complex)'%(s,) if s else '0'
    ret = TypedName(ret_values[-1].type, uid, alloc)
    func.declarations.append(ret)
    if s:
        func.expressions.append(mul_s_m('*'.join(_.name for _ in ret_values[:-1]),
                                        ret_values[-1].name,
                                        uid))
    else:
        func.expressions.append("%s = %s"%(uid, '*'.join(_.name for _ in ret_values)))
    func.return_value = ret
    return func

def dot(a, b, out):
    return '''# {out} = {a}.{b}
    ii = {a}.shape[0]
    jj = {b}.shape[1]
    kk = {b}.shape[0]
    for i in range(ii):
        for j in range(jj):
            c = 0
            for k in range(kk):
                c += {a}[i,k]*{b}[k,j]
            {out}[i,j] = c'''.format(
    out = out,
    a = a,
    b = b)
@_generate_cython.register(Dot)
def _(expr, func):
    if len(expr) != 2:
        raise NotImplementedError('Can not translate the dot products of more than two matrices %s to cython.'%expr)
    ret_values = [generate_cython(_, func).return_value for _ in expr]
    uid = next(func.uid)
    alloc = 'np.zeros(%s, dtype=np.complex)'%(shape(expr),) if shape(expr) else '0'
    ret = TypedName(ret_values[-1].type, uid, alloc)
    func.declarations.append(ret)
    exp = dot(ret_values[0].name, ret_values[1].name, ret.name)
    func.expressions.append(exp)
    func.return_value = ret
    return func

@_generate_cython.register(ScalarFunction)
def _(expr, func):
    uid = next(func.uid)
    arg = generate_cython(expr[0], func).return_value
    ret = TypedName('double', uid, '0')
    func.declarations.append(ret)
    func.expressions.append('%s = %s(%s)'%(
        uid,
        type(expr).__name__,
        arg.name
        ))
    func.return_value = ret
    return func


linear_ode_solver_template = """
ctypedef void rhs_t(double t, double *y, double *ydot)
cdef extern from "cvode_simple_interface.h":
    struct cvsi_instance:
        pass
    cvsi_instance *cvsi_setup(rhs_t *rhs,
                              double *y0, int neq,
                              double t0,
                              long mxsteps, double reltol, double abstol)
    int cvsi_step(cvsi_instance *instance, double t)
    void cvsi_destroy(cvsi_instance *instance)

cdef inline void RHS(double t, double *y, double *ydot):
    generated_function(t,
                       <double complex [:{neq}, :1]> <double complex *> y,
                       <double complex [:{neq}, :1]> <double complex *> ydot)
    # TODO The casts above need to allocate new memview structs each time
    # It is noticeable on 2x2 matrices (10%), not noticeable otherwise

cpdef list pythonsolve(
        np.ndarray[np.double_t, ndim=1] ts,
        np.ndarray[np.complex_t, ndim=2] y0,
        long mxsteps, double rtol, double atol):
    if ts[0] == 0:
        ts = ts[1:]
        res = [y0]
    else:
        res = []
    cdef np.ndarray[np.complex_t, ndim=2, mode="c"] _y0 = np.copy(y0)
    cdef int neq = np.size(_y0)*2
    cdef cvsi_instance *instance = cvsi_setup(RHS, <double *> &_y0[0,0], neq,
                                              0, mxsteps, rtol, atol)
    for t in ts:
        cvsi_step(instance, t)
        res.append(np.copy(_y0))
    cvsi_destroy(instance)

    return res
"""
class ODESolver(BaseCythonFunction):
    main_call = 'pythonsolve'
    def __str__(self):
        string = super(ODESolver, self).__str__()
        string += linear_ode_solver_template.format(
            return_value = self.return_value,
            neq = '%d'%dim(self.rev_memoized()[self.arguments[-1]])
            )
        return string
