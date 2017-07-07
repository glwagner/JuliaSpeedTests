# Functions for testing the speed of an fft.
#   - fftloop           : complex fft/ifft 
#   - realfftloop       : real rfft/irfft
#   - inplacefftloop    : 'inplace' complex fft/ifft
#   - fftloop_plan      : planned complex fft/ifft with pre-allocated output
#   - realfftloop_plan  : planned real rfft/irfft with pre-allocated output


function fftloop(a::Array{Complex128, 2}, ah::Array{Complex128, 2}, nloops::Int)
  for i = 1:nloops
    ah = fft(a)
    a = ifft(ah)
  end
end

function realfftloop(a::Array{Float64, 2}, ah::Array{Complex128, 2}, n::Int, nloops::Int)
  for i = 1:nloops
    ah  = rfft(a)
    a = irfft(ah, n)
  end
end

function inplacefftloop!(a::Array{Complex128, 2}, nloops::Int)
  for i = 1:nloops
    fft!(a)
    ifft!(a)
  end
end

function fftloop_plan(a::Array{Complex128, 2}, ah::Array{Complex128, 2}, 
  pfft::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,false,2},
  pifft::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,false,2},Float64},
  nloops::Int)

  for i = 1:nloops
    A_mul_B!(ah, pfft, a);
    A_mul_B!(a, pifft, ah);
  end
end

function realfftloop_plan(a::Array{Float64, 2}, ah::Array{Complex128, 2}, 
  prfft::Base.DFT.FFTW.rFFTWPlan{Float64,-1,false,2},
  pirfft::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,2},Float64},
  n::Int, nloops::Int)
  for i = 1:nloops
    A_mul_B!(ah, prfft, a)
    A_mul_B!(a, pirfft, ah)
  end
end

# Initialize random number generator
srand(123)
effort = FFTW.MEASURE
nloops = 400

for nthreads in [1, 2]
  for N in [32, 64, 128, 256, 512, 1024]

    FFTW.set_num_threads(nthreads)
    n = N

    a   = exp.(2*im*pi*rand(n, n))
    ah  = ifft(a)
    ar  = rand(n, n)
    arh = rfft(ar)

    pfft   = plan_fft(a; flags=effort);
    prfft  = plan_rfft(ar; flags=effort);
    pifft  = plan_ifft(ah; flags=effort);
    pirfft = plan_irfft(arh, n; flags=effort);
    
    # Compile
    fftloop(a, ah, 1)
    fftloop_plan(a, ah, pfft, pifft, 1)
    inplacefftloop!(a, 1)
    realfftloop(ar, arh, n, 1)
    realfftloop_plan(ar, arh, prfft, pirfft, n, 1)

    @printf "N: %5d^2, threads: %d, %24s : " n nthreads "out-of-place fft"
    @time fftloop(a, ah, nloops)

    @printf "N: %5d^2, threads: %d, %24s : " n nthreads "in-place planned fft"
    @time fftloop_plan(a, ah, pfft, pifft, nloops)

    @printf "N: %5d^2, threads: %d, %24s : " n nthreads "in-place fft"
    @time inplacefftloop!(a, nloops)

    @printf "N: %5d^2, threads: %d, %24s : " n nthreads "out-of-place real fft"
    @time realfftloop(ar, arh, n, nloops)

    @printf "N: %5d^2, threads: %d, %24s : " n nthreads "in-place planned rfft"
    @time realfftloop_plan(ar, arh, prfft, pirfft, n, nloops)

    println()

  end

  println()

end
