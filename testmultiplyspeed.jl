# Functions for testing the speed of elementwise array multiplications.

# ----------------------------------------------------------------------------- 
function mult3d_unroll!(n::Int,
  a::Array{Complex128, 3}, 
  b::Array{Complex128, 3}, 
  c::Array{Complex128, 3}, 
  nloops::Int)

  for loop = 1:nloops
    for j = 1:n
      @simd for i = 1:n
        @fastmath @inbounds c[i, j, 1] = a[i, j, 1]*b[i, j, 1]
      end
    end

    for j = 1:n
      @simd for i = 1:n
        @fastmath @inbounds c[i, j, 2] = a[i, j, 2]*b[i, j, 2]
      end
    end
  end

end


function mult3d_rolled!(n::Int, m::Int,
  a::Array{Complex128, 3}, 
  b::Array{Complex128, 3}, 
  c::Array{Complex128, 3}, 
  nloops::Int)

  for loop = 1:nloops
    for k = 1:m
      for j = 1:n
        @simd for i = 1:n
          @fastmath @inbounds c[i, j, k] = a[i, j, k]*b[i, j, k]
        end
      end
    end
  end

end


function mult!(n::Int,
  a::Array{Complex128, 2}, 
  b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, 
  nloops::Int)

  for loop = 1:nloops
    for j = 1:n, i = 1:n
      c[i, j] = a[i, j]*b[i, j]
    end
  end

end

function elemmult!(
  a::Array{Complex128, 3}, 
  b::Array{Complex128, 3}, 
  c::Array{Complex128, 3}, 
  nloops::Int)

  for loop = 1:nloops
    @fastmath c .= a.*b
  end

end



function elemmult!(
  a::Array{Complex128, 2}, 
  b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, 
  nloops::Int)

  for loop = 1:nloops
    c .= a.*b
  end

end


function threadedmult!(n::Int,
  a::Array{Complex128, 2}, 
  b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, 
  nloops::Int)

  for loop = 1:nloops
    Threads.@threads for j = 1:n
      for i = 1:n
        @inbounds c[i, j] = a[i, j]*b[i, j]
      end
    end
  end

end


function fastmult!(n::Int,
  a::Array{Complex128, 2}, 
  b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, 
  nloops::Int)

  for loop = 1:nloops
    for j = 1:n
      @simd for i = 1:n
        @fastmath @inbounds c[i, j] = a[i, j]*b[i, j]
      end
    end
  end

end

# Test
nloops = 1000
srand(123)

nthreads_julia = ENV["JULIA_NUM_THREADS"]
nthreads_mkl = ENV["MKL_NUM_THREADS"]

for n in [32, 64, 128, 256, 512, 1024]

  a  = exp.(2*im*pi*rand(n, n))
  b  = exp.(2*im*pi*rand(n, n))
  c  = exp.(2*im*pi*rand(n, n))

  m = 1
  a3 = exp.(2*im*pi*rand(n, n, m))
  b3 = exp.(2*im*pi*rand(n, n, m))
  c3 = exp.(2*im*pi*rand(n, n, m))

  # Compilation calls
  mult!(n, a, b, c, nloops)
  elemmult!(a, b, c, nloops)
  elemmult!(a3, b3, c3, nloops)
  fastmult!(n, a, b, c, nloops)
  threadedmult!(n, a, b, c, nloops)
  mult3d_rolled!(n, m, a3, b3, c3, nloops)
  #mult3d_unroll!(n, a3, b3, c3, nloops)
  #accmult!(n, a, b, c, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "loop"
  #@time mult!(n, a, b, c, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_mkl n "elem"
  #@time elemmult!(a, b, c, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "fast"
  #@time fastmult!(n, a, b, c, nloops)

  @printf "nthreads: %s, N: %5d^2: %8s:" nthreads_mkl n "elem"
  @time elemmult!(a3, b3, c3, nloops)

  @printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "fast"
  @time fastmult!(n, a, b, c, nloops)

  @printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "rolled"
  @time mult3d_rolled!(n, m, a3, b3, c3, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "unroll"
  #@time mult3d_unroll!(n, a3, b3, c3, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "threaded"
  #@time threadedmult!(n, a, b, c, nloops)

  #@printf "nthreads: %s, N: %5d^2: %8s:" nthreads_julia n "acc"
  #@time accmult!(n, a, b, c, nloops)

end
