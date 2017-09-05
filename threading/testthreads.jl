include("./algebras.jl")

using BenchmarkTools
using AlgebraTests

nloops = 10
nz = [64, 128, 256, 512, 1024, 2048, 4096]

# Compile step
n = 256
x, y, z = zeros(Float64, n, n), zeros(Float64, n, n), zeros(Float64, n, n)
a, b, c, d = rand(n, n), rand(n, n), rand(n, n), rand(n, n)
algebra_basic!(x, y, z, nloops, a, b, c, d) 
algebra_halffused!(x, y, z, nloops, a, b, c, d) 
algebra_unfused!(x, y, z, nloops, a, b, c, d) 
algebra_threaded!(x, y, z, nloops, a, b, c, d)


# ----------------------------------------------------------------------------- 
@printf("Testing basic and threaded algebra with %d threads...\n", 
  Threads.nthreads())
for n in nz

  x, y, z = zeros(Float64, n, n), zeros(Float64, n, n), zeros(Float64, n, n)
  a, b, c, d = rand(n, n), rand(n, n), rand(n, n), rand(n, n)
  
  @printf("n: %d\n", n)
  @printf("Basic:      ")
  @btime algebra_basic!($x, $y, $z, $nloops, $a, $b, $c, $d)

  @printf("Half-fused: ")
  @btime algebra_halffused!($x, $y, $z, $nloops, $a, $b, $c, $d)

  @printf("Unfused:    ")
  @btime algebra_unfused!($x, $y, $z, $nloops, $a, $b, $c, $d)

  @printf("Threaded:   ")
  @btime algebra_threaded!($x, $y, $z, $nloops, $a, $b, $c, $d)

  @printf("\n")

end
