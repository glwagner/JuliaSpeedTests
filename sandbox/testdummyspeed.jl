include("dummyproblem.jl")

using DummyProblem

# Parameters
nthreads = 2
nsteps = 1000
nx = 256

# Initialize
g = init_grid(nx)
v = init_vars(g)

# Prepare FFT
FFTW.set_num_threads(nthreads)
plan_fft(v.a, flags=FFTW.MEASURE)
plan_ifft(v.ah, flags=FFTW.MEASURE)

# Compile step
step_nsteps!(1, v, g)

@time step_nsteps!(nsteps, v, g)
