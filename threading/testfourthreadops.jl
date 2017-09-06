include("./fourthreadops.jl")

using BenchmarkTools
using FourThreadOps

nloops = 100
nz = [64, 128, 256, 512, 1024, 2048, 4096]

# Compile!
n = 16
s, v = Solution(n), Vars(n)

basic!(s, nloops, v)
acc!(s, nloops, v)
threaded!(s, nloops, v)


@printf("\nTesting basic and threaded algebra with %d threads...\n", 
  Threads.nthreads())

for n in nz

  s, v = Solution(n), Vars(n)

  # Warm up
  basic!(s, nloops, v)
  acc!(s, nloops, v)
  threaded!(s, nloops, v)

  # Time
  @printf("n: %d\n", n)
  @printf("Basic:    ")
  @btime basic!($s, $nloops, $v)

  @printf("Acc:      ")
  @btime acc!($s, $nloops, $v)

  @printf("Threaded: ")
  @btime threaded!($s, $nloops, $v)

  @printf("\n")

end
