module FourThreadOps

using ParallelAccelerator

export Solution, Vars
export basic!, acc!, threaded!


# ----------------------------------------------------------------------------- 
# Types 
type Solution
  w::Array{Float64, 2}
  x::Array{Float64, 2}
  y::Array{Float64, 2}
  z::Array{Float64, 2}
end

function Solution(n::Int)
  w = zeros(n, n)
  x = zeros(n, n)
  y = zeros(n, n)
  z = zeros(n, n)
  Solution(w, x, y, z)
end

type Vars
  a::Array{Float64, 2}
  b::Array{Float64, 2}
  c::Array{Float64, 2}
  d::Array{Float64, 2}
end

function Vars(n::Int)
  a = rand(n, n)
  b = rand(n, n)
  c = rand(n, n)
  d = rand(n, n)
  Vars(a, b, c, d)
end


# ----------------------------------------------------------------------------- 
# Ops
function basic!(s::Solution, nloops::Int, v::Vars)
  for loop = 1:nloops
    @. s.w = v.a*v.b*v.c*v.d
    @. s.x = v.a*v.b + v.c*v.d
    @. s.y = v.a - 2.0*v.b*v.c*v.d
    @. s.z = v.a^2.0 + v.b^2.0 - v.c*v.d
  end

  nothing
end 

@acc function acc!(s::Solution, nloops::Int, v::Vars)
  for loop = 1:nloops
    @. s.w = v.a*v.b*v.c*v.d
    @. s.x = v.a*v.b + v.c*v.d
    @. s.y = v.a - 2.0*v.b*v.c*v.d
    @. s.z = v.a^2.0 + v.b^2.0 - v.c*v.d
  end

  nothing
end 


function threaded!(s::Solution, nloops::Int, v::Vars)

  exprs = [
    :(@. s.w = v.a*v.b*v.c*v.d), 
    :(@. s.x = v.a*v.b + v.c*v.d),
    :(@. s.y = v.a - 2.0*v.b*v.c*v.d), 
    :(@. s.z = v.a^2.0 + v.b^2.0 - v.c*v.d)
  ]

  for loop = 1:nloops
    Threads.@threads for expr in exprs
      eval(expr)
    end
  end

  nothing
end 


end
