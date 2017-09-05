# Functions for testing the speed of elementwise array multiplications.

function fusedmult!(
  a::Array{Complex{Float64}, 2}, 
  b::Array{Complex{Float64}, 2}, 
  c::Array{Complex{Float64}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(
  a::Array{Float64, 2}, 
  b::Array{Complex{Float64}, 2}, 
  c::Array{Complex{Float64}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(
  a::Array{Complex{Float32}, 2}, 
  b::Array{Complex{Float32}, 2}, 
  c::Array{Complex{Float32}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(
  a::Array{Float32, 2}, 
  b::Array{Complex{Float32}, 2}, 
  c::Array{Complex{Float32}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(
  a::Array{Complex{Float16}, 2}, 
  b::Array{Complex{Float16}, 2}, 
  c::Array{Complex{Float16}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end

function fusedmult!(
  a::Array{Float16, 2}, 
  b::Array{Complex{Float16}, 2}, 
  c::Array{Complex{Float16}, 2}, nloops::Int)
  for loop = 1:nloops
    @fastmath c .= a.*b
  end
end




# Test
nloops = 1000
srand(123)

for ntimes in 1:4

println()

for n in [32, 64, 128, 256, 512]

  ac = exp.(2*im*pi*rand(n, n))
  bc = exp.(2*im*pi*rand(n, n))
  cc = exp.(2*im*pi*rand(n, n))

  ar = real.(ac)

  # Compilation calls
  fusedmult!(ac, bc, cc, nloops)
  fusedmult!(ar, bc, cc, nloops)

  @printf "N: %5d^2: %20s:" n "fused"
  @time fusedmult!(ac, bc, cc, nloops)

  @printf "N: %5d^2: %20s:" n "fused w promotion"
  @time fusedmult!(ar, bc, cc, nloops)

  println()

end

end
