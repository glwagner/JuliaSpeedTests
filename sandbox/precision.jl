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
  a::Array{Complex{Float32}, 2}, 
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



# Test
nloops = 1000
srand(123)

for ntimes in 1:4

println()

for n in [32, 64, 128, 256, 512]

  ac64 = exp.(2*im*pi*rand(n, n))
  bc64 = exp.(2*im*pi*rand(n, n))
  cc64 = exp.(2*im*pi*rand(n, n))

  ac32 = convert(Array{Complex{Float32}}, ac64)
  bc32 = convert(Array{Complex{Float32}}, bc64)
  cc32 = convert(Array{Complex{Float32}}, cc64)

  ac16 = convert(Array{Complex{Float16}}, ac64)
  bc16 = convert(Array{Complex{Float16}}, bc64)
  cc16 = convert(Array{Complex{Float16}}, cc64)

  # Compilation calls
  fusedmult!(ac64, bc64, cc64, nloops)
  fusedmult!(ac32, bc32, cc32, nloops)
  fusedmult!(ac16, bc16, cc16, nloops)

  @printf "N: %5d^2: %8s:" n "fused 64"
  @time fusedmult!(ac64, bc64, cc64, nloops)

  @printf "N: %5d^2: %8s:" n "fused 32"
  @time fusedmult!(ac32, bc32, cc32, nloops)

  #@printf "N: %5d^2: %8s:" n "fused 16"
  #@time fusedmult!(ac16, bc16, cc16, nloops)

  println()

end

end
