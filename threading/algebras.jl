module AlgebraTests

using ParallelAccelerator

export algebra_basic!, algebra_threaded!, algebra_unfused!,
       algebra_halffused!, algebra_acc!, algebra_threadfuse!


function algebra_basic!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})
  
  for loop = 1:nloops
    @. x = a*b + c*d
    @. y = a - 2.0*b*c*d
    @. z = a^2.0 + b^2.0 - c*d
  end

  nothing
end 


@acc function algebra_acc!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})
  
  for loop = 1:nloops
    @. x = a*b + c*d
    @. y = a - 2.0*b*c*d
    @. z = a^2.0 + b^2.0 - c*d
  end

  nothing
end 




function algebra_halffused!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})

  n, m = size(x)

  for loop = 1:nloops
    for j in 1:m 
      @views @. x[:, j] = a[:, j]*b[:, j] + c[:, j]*d[:, j]
    end

    for j in 1:m 
      @views @. y[:, j] = a[:, j] - 2.0*b[:, j]*c[:, j]*d[:, j]
    end

    for j in 1:m 
      @views @. z[:, j] = a[:, j]^2.0 + b[:, j]^2.0 - c[:, j]*d[:, j]
    end

  end

  nothing
end 





function algebra_unfused!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})

  n, m = size(x)

  for loop = 1:nloops
    for j in 1:m 
      @simd for i in 1:n
        @fastmath @inbounds x[i, j] = a[i, j]*b[i, j] + c[i, j]*d[i, j]
      end
    end

    for j in 1:m 
      @simd for i in 1:n
        @fastmath @inbounds y[i, j] = a[i, j] - 2.0*b[i, j]*c[i, j]*d[i, j]
      end
    end

    for j in 1:m 
      @simd for i in 1:n
        @fastmath @inbounds z[i, j] = a[i, j]^2.0 + b[i, j]^2.0 - c[i, j]*d[i, j]
      end
    end

  end

  nothing
end 




function algebra_threaded!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})

  n, m = size(x)

  for loop = 1:nloops
    Threads.@threads for j in 1:m 
      @simd for i in 1:n
        @fastmath @inbounds x[i, j] = a[i, j]*b[i, j] + c[i, j]*d[i, j]
      end
      @simd for i in 1:n
        @fastmath @inbounds y[i, j] = a[i, j] - 2.0*b[i, j]*c[i, j]*d[i, j]
      end
      @simd for i in 1:n
        @fastmath @inbounds z[i, j] = a[i, j]^2.0 + b[i, j]^2.0 - c[i, j]*d[i, j]
      end
    end
  end

  nothing
end 


function algebra_threadfuse!(x::Array{Float64, 2}, y::Array{Float64, 2}, 
  z::Array{Float64, 2}, nloops::Int, a::Array{Float64, 2}, 
  b::Array{Float64, 2}, c::Array{Float64, 2}, d::Array{Float64, 2})

  n, m = size(x)

  for loop = 1:nloops
    Threads.@threads for j in 1:m 
      @views @. x[:, j] = a[:, j]*b[:, j] + c[:, j]*d[:, j]
      @views @. y[:, j] = a[:, j] - 2.0*b[:, j]*c[:, j]*d[:, j]
      @views @. z[:, j] = a[:, j]^2.0 + b[:, j]^2.0 - c[:, j]*d[:, j]
    end
  end

  nothing
end 




end
