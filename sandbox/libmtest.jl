using Base.Threads

println("Number of threads = $(nthreads())")

# const libm_name = "libopenlibm"
const libm_name = "libm"

@show libm_name

sin1(x::Float64) = ccall((:sin, libm_name), Float64, (Float64,), x)
cos1(x::Float64) = ccall((:cos, libm_name), Float64, (Float64,), x)

@noinline function test1!(y, x)
    # @assert length(y) == length(x)
    for i = 1:length(x)
        y[i] = sin1(x[i])^2 + cos1(x[i])^2
    end
    y
end

@noinline function testn!(y::Vector{Float64}, x::Vector{Float64})
    # @assert length(y) == length(x)
    Threads.@threads for i = 1:length(x)
        y[i] = sin1(x[i])^2 + cos1(x[i])^2
    end
    y
end

function run_tests()
    n = 10^7
    x = rand(n)
    y = zeros(n)
    test1!(y, x)
    testn!(y, x)
    @time for i in 1:10
        test1!(y, x)
    end
    @time for i in 1:10
        testn!(y, x)
    end
end

run_tests()
