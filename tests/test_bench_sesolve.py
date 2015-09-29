# Based on some of the example notebooks.
from cutiepy import *
import pytest

class TestSesolve:
    @pytest.mark.slow
    def test_compile(self, benchmark):
        initial_state = basis(2, 0)
        ω0 = 1
        Δ = 0.000
        Ω = 0.005
        ts = 2*np.pi/Ω*np.linspace(0,1,40)
        H = ω0/2 * sigmaz() + Ω * sigmax() * sin((ω0+Δ)*t)
        def sesolve_no_memoization(*args):
            r = sesolve(*args)
            cutiepy.codegen._expr_to_func_memoized = {}
            return r
        rabi_args = H, initial_state, ts
        benchmark(sesolve_no_memoization, *rabi_args)

    def test_rabi(self, benchmark):
        initial_state = basis(2, 0)
        ω0 = 1
        Δ = 0.000
        Ω = 0.005
        ts = 2*np.pi/Ω*np.linspace(0,1,40)
        H = ω0/2 * sigmaz() + Ω * sigmax() * sin((ω0+Δ)*t)
        rabi_args = H, initial_state, ts
        benchmark(sesolve, *rabi_args)

    def test_rabi_rwa(self, benchmark):
        initial_state = basis(2, 0)
        ω0 = 1
        Δ = 0.002
        Ω = 0.005
        ts = 2*np.pi/Ω*np.linspace(0,1,40)
        Hp = Δ/2 * sigmaz() + Ω/2 * sigmax()
        benchmark(sesolve, Hp, initial_state, ts)

    def test_bench_coherent_in_harm_oscillator(self, benchmark):
        N_cutoff = 40
        α = 2.5
        initial_state = coherent(N_cutoff, α)
        H = num(N_cutoff)
        ts = 2*np.pi*np.linspace(0,1,41)
        benchmark(sesolve, H, initial_state, ts)

    def test_bench_jc_revival(self, benchmark):
        ω = 1
        g = 0.1
        ts = np.pi/g*np.linspace(0,1,150)
        N_cutoff = 50
        H0 = ω*(tensor(num(N_cutoff), identity(2)) + 0.5 * tensor(identity(N_cutoff), sigmaz()))
        Hp = g*(tensor(destroy(N_cutoff),sigmap()) + tensor(create(N_cutoff), sigmam()))
        alpha = 5
        coh = tensor(coherent(N_cutoff, alpha), basis(2,0))
        ts = 1/g*np.linspace(0,1,200)
        benchmark(sesolve, H0 + Hp, coh, ts)
