# Functions for testing the speed of elementwise array multiplications.
using ParallelAccelerator

# ----------------------------------------------------------------------------- 
function mult3d_unroll!(n::Int, a::Array{Complex128, 3},
  b::Array{Complex128, 3}, c::Array{Complex128, 3}, nloops::Int)
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


@acc function accmult!(a::Array{Complex128, 3}, b::Array{Complex128, 3}, 
  c::Array{Complex128, 3}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

@acc function accmult!(a::Array{Complex128, 2}, b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(a::Array{Complex128, 3}, b::Array{Complex128, 3}, 
  c::Array{Complex128, 3}, nloops::Int)
  for loop = 1:nloops 
    @fastmath c .= a.*b
  end
end

function fusedmult!(a::Array{Complex128, 2}, b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function slowloopmult!(n::Int, a::Array{Complex128, 2}, b::Array{Complex128, 2}, 
  c::Array{Complex128, 2}, nloops::Int)
  for loop = 1:nloops, j = 1:n, i = 1:n
    c[i, j] = a[i, j]*b[i, j]
  end
end

function loopmult!(n::Int, m::Int, a::Array{Complex128, 3}, 
  b::Array{Complex128, 3}, c::Array{Complex128, 3}, nloops::Int)
  for loop = 1:nloops, k = 1:m, j = 1:n
    @simd for i = 1:n
      @fastmath @inbounds c[i, j, k] = a[i, j, k]*b[i, j, k]
    end
  end
end

function parloopmult!(n::Int, m::Int, a::Array{Complex128, 3}, 
  b::Array{Complex128, 3}, c::Array{Complex128, 3}, nloops::Int)
  for loop = 1:nloops, k = 1:m
    @parallel for j = 1:n
      @simd for i = 1:n
        @fastmath @inbounds c[i, j, k] = a[i, j, k]*b[i, j, k]
      end
    end
  end
end

function loopmult!(n::Int, a::Array{Complex128, 2},  b::Array{Complex128, 2},  
  c::Array{Complex128, 2}, nloops::Int)
  for loop = 1:nloops, j = 1:n
    @simd for i = 1:n
      @fastmath @inbounds c[i, j] = a[i, j]*b[i, j]
    end
  end
end

function parloopmult!(n::Int, a::Array{Complex128, 2},  b::Array{Complex128, 2},  
  c::Array{Complex128, 2}, nloops::Int)
  for loop = 1:nloops
    @parallel for j = 1:n
      @simd for i = 1:n
        @fastmath @inbounds c[i, j] = a[i, j]*b[i, j]
      end
    end
  end
end




# Test
nloops = 1000
srand(123)


for n in [32, 64, 128, 256, 512, 1024]

  a2 = exp.(2*im*pi*rand(n, n))
  b2 = exp.(2*im*pi*rand(n, n))
  c2 = exp.(2*im*pi*rand(n, n))

  m = 1
  a3 = exp.(2*im*pi*rand(n, n, m))
  b3 = exp.(2*im*pi*rand(n, n, m))
  c3 = exp.(2*im*pi*rand(n, n, m))

  #a2s = SharedArray{Complex128, 2}((n, n), 
  #  init=a2s->a2s[Base.localindexes(a2s)]=myid())
  #b2s = SharedArray{Complex128, 2}((n, n), 
  #  init=b2s->b2s[Base.localindexes(b2s)]=myid())
  #c2s = SharedArray{Complex128, 2}((n, n), 
  #  init=c2s->c2s[Base.localindexes(c2s)]=myid())

  # Compilation calls
  loopmult!(n, a2, b2, c2, nloops)
  loopmult!(n, m, a3, b3, c3, nloops)
  fusedmult!(a2, b2, c2, nloops)
  fusedmult!(a3, b3, c3, nloops)
  accmult!(a2, b2, c2, nloops)
  accmult!(a3, b3, c3, nloops)

  if m == 2 
    mult3d_unroll!(n, a3, b3, c3, nloops)
  end

  @printf "N: %5d^2: %8s:" n "fused 2"
  @time fusedmult!(a2, b2, c2, nloops)

  @printf "N: %5d^2: %8s:" n "fused 3"
  @time fusedmult!(a3, b3, c3, nloops)

  @printf "N: %5d^2: %8s:" n "acc 2"
  @time accmult!(a2, b2, c2, nloops)

  @printf "N: %5d^2: %8s:" n "acc 3"
  @time accmult!(a3, b3, c3, nloops)

  @printf "N: %5d^2: %8s:" n "loop 2"
  @time loopmult!(n, a2, b2, c2, nloops)

  @printf "N: %5d^2: %8s:" n "loop 3"
  @time loopmult!(n, m, a3, b3, c3, nloops)

end
