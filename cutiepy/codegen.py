'''
Code generation module
======================
'''

import collections
import functools
import numbers
import tempfile

import numpy as np
from scipy.sparse import csr_matrix

import cutiepy
from .symbolic import (Scalar, NotScalar, ScalarFunction, Add, Mul, Pow, Dot,
        TensorProd, shape, isnumerical, isscalar, _CG_AppliedLindbladSuperoperator)


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
from scipy.linalg.cython_blas cimport zgemv, zgemm

cdef int iONE=1, iZERO=0
cdef double complex zONE=1, zZERO=0, zNHALF=-0.5

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
    def __init__(self):
        self.uid = uid()
        self.arguments = []
        self.intermediate_results = []
        self.expressions = []
        self.return_value = None
        self.memoized = {}
        self.predefined_constants = []
        self.predefined_constants_num = []
        self._compiled_mod = None

    def globals_declarations_s(self):
        s  = '\n'.join('cdef {0.type} {0.name} = {0.allocate}'.format(_)
                       for _ in self.intermediate_results[:-1])
        s += '\n'
        s += '\n'.join('cdef {0.type} {0.name} # initialized in setup'.format(_)
                       for _ in self.predefined_constants)
        return s if s else '# no global declarations'
    def intermediate_results_globals_s(self):
        s = ', '.join(_.name for _ in self.intermediate_results[:-1])
        return 'global %s'%s if s else '# no intermediate results globals'
    def predefined_constants_globals_s(self):
        s = ', '.join(_.name for _ in self.predefined_constants)
        return 'global %s'%s if s else '# no predefined constants globals'
    def arguments_s(self):
        s = ',\n    '.join('{0.type} {0.name}'.format(_) for _ in self.arguments)
        return s if s else '    # no arguments'
    def expressions_s(self):
        s = '\n'.join('    %s'%_ for _ in self.expressions)
        return s if s else '    # no calculations'
    def setup_arguments_s(self):
        s = ',\n    '.join('{0.type} _{0.name}'.format(_) for _ in
                           self.predefined_constants)
        return s if s else '    # no arguments'
    def setup_expressions_s(self):
        s = '\n'.join('    {0.name} = _{0.name}'.format(_) for _ in
                self.predefined_constants)
        return s if s else '    pass # no setup'

    def argument_order(self, args):
        new_constants = list(set(self.arguments)-{self.memoized[_] for _ in args if _ in self.memoized})
        self.predefined_constants += new_constants
        self.predefined_constants_num += [self.rev_memoized()[_].numerical for _ in new_constants]
        def get_typedname(a):
            if a in self.memoized:
                pass
            elif isinstance(a, Scalar):
                self.memoized[a] = TypedName('double', next(self.uid), '0')
            else:
                raise NotImplementedError('Unused nonscalar argument: %s'%a)
            return self.memoized[a]
        self.arguments = [get_typedname(_) for _ in args]
    def rev_memoized(self):
        return dict(zip(self.memoized.values(), self.memoized.keys()))

    def __str__(self):
        return base_cython_function_template.format(
                   globals_declarations = self.globals_declarations_s(),
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
                self._extension = Extension(extname,
                              sources=[filename],
                              include_dirs=[os.path.join(os.path.dirname(cutiepy.__file__), 'include')],
                              libraries=["sundials_cvode", "sundials_nvecserial"],
                              extra_compile_args=["-O3"]
                            )
                try:
                    module_path = pyx_to_dll(filename, ext=self._extension)
                except Exception as e:
                    raise e
                finally:
                    os.remove(os.path.join(folder, extname+'.c'))
            self._compiled_mod = imp.load_dynamic(filename.split('.')[0],module_path)
        self._compiled_mod.setup_generated_function(*self.predefined_constants_num)
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
    uid = next(func.uid)
    if isnumerical(expr) and isinstance(expr.numerical, csr_matrix): # can only be a predefined constant
        ret = TypedName('double complex [:]', uid, None)
        func.predefined_constants.append(ret)
        func.predefined_constants.append(TypedName('int [:]', uid+'_pointers', None))
        func.predefined_constants.append(TypedName('int [:]', uid+'_indices', None))
        num = expr.numerical
        func.predefined_constants_num.extend([num.data, num.indptr, num.indices])
        func.return_value = ret
    else: # can be both and argument or a predefined constant
        ret = TypedName('double complex [:, :]', uid, 'np.empty(%s, dtype=np.complex)'%(shape(expr),))
        func.arguments.append(ret)
        func.return_value = ret
    return func

@_generate_cython.register(Pow)
def _(expr, func):
    if isscalar(expr):
        uid = next(func.uid)
        arg =   generate_cython(expr[0], func).return_value
        power = generate_cython(expr[1], func).return_value
        ret = TypedName('double', uid, '0')
        func.intermediate_results.append(ret)
        func.expressions.append('%s = (%s)**(%s)'%(
            uid,
            arg.name,
            power.name
            ))
        func.return_value = ret
        return func
    else: # TODO do better by nesting squareing  of matrices
        return _generate_cython(super(Dot, Dot).__new__(Dot, *[expr[0]]*expr[1]), func) # TODO encapsulate the "use super to skip canonicalization" trick

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
    alloc = 'np.empty(%s, dtype=np.complex)'%(s,) if s else '0'
    ret = TypedName(ret_values[-1].type, uid, alloc)
    func.intermediate_results.append(ret)
    if s:
        func.expressions.append(variadic_add_m(ret_values, ret))
    else:
        func.expressions.append("%s = %s"%(uid, '+'.join(_.name for _ in ret_values)))
    func.return_value = ret
    return func

def mul_s_m(ss, m, out):
    return '''# {out} = {s}*{m}
    ii = {m}.shape[0]
    jj = {m}.shape[1]
    for i in range(ii):
        for j in range(jj):
            {out}[i,j] = {s}*{m}[i,j]'''.format(
    out = out.name,
    s = '*'.join(_.name for _ in ss),
    m = m.name)
@_generate_cython.register(Mul)
def _(expr, func):
    ret_values = [generate_cython(_, func).return_value for _ in expr]
    uid = next(func.uid)
    s = shape(expr)
    alloc = 'np.empty(%s, dtype=np.complex)'%(s,) if s else '0'
    ret = TypedName(ret_values[-1].type, uid, alloc)
    func.intermediate_results.append(ret)
    if s:
        func.expressions.append(mul_s_m(ret_values[:-1],
                                        ret_values[-1],
                                        ret))
    else:
        func.expressions.append("%s = %s"%(uid, '*'.join(_.name for _ in ret_values)))
    func.return_value = ret
    return func

def dot(a, b, out):
    return '''# {out} = {a}.{b}
    ii = {a}.shape[0]
    jj = {a}.shape[1]
    kk = {b}.shape[1]
    zgemm('N', 'N', &kk, &ii, &jj, &zONE, &{b}[0,0], &kk, &{a}[0,0], &jj, &zZERO, &{out}[0,0], &kk)'''.format(
    # Commented out col-major order (BLAS default):
    #zgemm('N', 'N', &ii, &kk, &jj, &zONE, &{a}[0,0], &ii, &{b}[0,0], &jj, &zZERO, &{out}[0,0], &ii)
    out = out.name,
    a = a.name,
    b = b.name)
def dot_sparse(csr, vec, out):
    return '''# {out} = {csr}.{vec}
    ii = {csr}_pointers.shape[0]-1
    for i in range(ii):
        jj = {csr}_pointers[i]
        kk = {csr}_pointers[i+1]
        c = 0
        for j in range(jj,kk):
            k = {csr}_indices[j]
            c += {csr}[j]*{vec}[k,0]
        {out}[i,0] = c
    '''.format(
    out = out.name,
    csr = csr.name,
    vec = vec.name)
@_generate_cython.register(Dot)
def _(expr, func):
    if len(expr) != 2: # TODO verify this is the optimal dot product (document when it is and when it is not)
        mat_expr, vec_expr = expr[0], Dot(*expr[1:])
    else:
        mat_expr, vec_expr = expr
    mat = generate_cython(mat_expr, func).return_value
    vec = generate_cython(vec_expr, func).return_value
    alloc = 'np.empty(%s, dtype=np.complex)'%(shape(expr),) if shape(expr) else '0'
    uid = next(func.uid)
    ret = TypedName(vec.type, uid, alloc)
    func.intermediate_results.append(ret)
    func.return_value = ret
    if isnumerical(mat_expr) and isinstance(expr[0].numerical, csr_matrix):
        func.expressions.append(dot_sparse(mat, vec, ret))
    else:
        func.expressions.append(dot(mat, vec, ret))
    return func

def applied_lindblad(c_op, rho, out):
    return '''# {out} = Lindblad({c_op}).{rho}
    ii = {out}.shape[0]
    # tmp = c.rho
    zgemm('N', 'N', &ii, &ii, &ii, &zONE, &{rho}[0,0], &ii, &{c_op}[0,0], &ii, &zZERO, &{out}_tmp[0,0], &ii)
    # out = tmp.c'  # out = c.rho.c'
    zgemm('C', 'N', &ii, &ii, &ii, &zONE, &{c_op}[0,0], &ii, &{out}_tmp[0,0], &ii, &zZERO, &{out}[0,0], &ii)
    # tmp = c'.c
    zgemm('N', 'C', &ii, &ii, &ii, &zONE, &{c_op}[0,0], &ii, &{c_op}[0,0], &ii, &zZERO, &{out}_tmp[0,0], &ii)
    # out += rho.tmp += tml.rho
    zgemm('N', 'N', &ii, &ii, &ii, &zNHALF, &{rho}[0,0], &ii, &{out}_tmp[0,0], &ii, &zONE, &{out}[0,0], &ii)
    zgemm('N', 'N', &ii, &ii, &ii, &zNHALF, &{out}_tmp[0,0], &ii, &{rho}[0,0], &ii, &zONE, &{out}[0,0], &ii)
    '''.format(
    out = out.name,
    c_op = c_op.name,
    rho = rho.name)
@_generate_cython.register(_CG_AppliedLindbladSuperoperator)
def _(expr, func):
    c_op = generate_cython(expr[0], func).return_value
    rho  = generate_cython(expr[1], func).return_value
    alloc = 'np.empty(%s, dtype=np.complex)'%(shape(expr),)
    uid = next(func.uid)
    ret_tmp = TypedName(rho.type, uid+'_tmp', alloc)
    ret = TypedName(rho.type, uid, alloc)
    func.intermediate_results.append(ret_tmp)
    func.intermediate_results.append(ret)
    func.return_value = ret
    func.expressions.append(applied_lindblad(c_op,rho,ret))
    return func

@_generate_cython.register(ScalarFunction)
def _(expr, func):
    uid = next(func.uid)
    arg = generate_cython(expr[0], func).return_value
    ret = TypedName('double', uid, '0')
    func.intermediate_results.append(ret)
    func.expressions.append('%s = %s(%s)'%(
        uid,
        type(expr).__name__,
        arg.name
        ))
    func.return_value = ret
    return func


ndarray_function_template = """
cpdef np.ndarray[np.complex_t, ndim=2] pythoncall({args}):
    cdef np.ndarray[np.complex_t, ndim=2] result = np.empty({result_shape}, complex)
    generated_function({args}, result)
    return result
"""
class NDArrayFunction(BaseCythonFunction):
    def __str__(self):
        string = super(NDArrayFunction, self).__str__()
        string += ndarray_function_template.format(
            result_shape = str(shape(self.rev_memoized()[self.return_value])),
            args = ','.join('_%d'%i for i in range(len(self.arguments)))
            )
        return string


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
                       <double complex [:{shape[0]}, :{shape[1]}]> <double complex *> y,
                       <double complex [:{shape[0]}, :{shape[1]}]> <double complex *> ydot)
    # TODO The casts above need to allocate new memview structs each time
    # It is noticeable on 2x2 matrices (10%), not noticeable otherwise

cpdef list pythonsolve(
        np.ndarray[np.double_t, ndim=1] ts,
        np.ndarray[np.complex_t, ndim=2] y0,
        long mxsteps, double rtol, double atol,
        progressbar):
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
        progressbar.step()
    cvsi_destroy(instance)

    progressbar.stop()
    return res
"""
class ODESolver(BaseCythonFunction):
    def __str__(self):
        string = super(ODESolver, self).__str__()
        string += linear_ode_solver_template.format(
            shape = shape(self.rev_memoized()[self.arguments[-1]])
            )
        return string
