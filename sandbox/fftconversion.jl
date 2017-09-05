# Functions for testing the speed of an fft.
#   - fftloop           : complex fft/ifft 
#   - realfftloop       : real rfft/irfft
#   - inplacefftloop    : 'inplace' complex fft/ifft
#   - fftloop      : planned complex fft/ifft with pre-allocated output
#   - realfftloop  : planned real rfft/irfft with pre-allocated output


function fftloop(a::Array{Complex128, 2}, ah::Array{Complex128, 2}, 
  pfft::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,false,2},
  pifft::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,false,2},Float64},
  nloops::Int)

  for i = 1:nloops
    A_mul_B!(ah, pfft, a);
    A_mul_B!(a, pifft, ah);
  end
end


function realfftloop(a::Array{Float64, 2}, ah::Array{Complex128, 2}, 
  prfft::Base.DFT.FFTW.rFFTWPlan{Float64,-1,false,2},
  pirfft::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,2},Float64},
  nloops::Int)
  for i = 1:nloops
    A_mul_B!(ah, prfft, a)
    A_mul_B!(a, pirfft, ah)
  end
end

function realfftloop(a::Array{Complex{Float64}, 2}, ar::Array{Float64, 2}, 
  ah::Array{Complex128, 2}, 
  prfft::Base.DFT.FFTW.rFFTWPlan{Float64,-1,false,2},
  pirfft::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,2},Float64},
  nloops::Int)
  for i = 1:nloops
    ar .= real.(a)
    A_mul_B!(ah, prfft, ar)
    A_mul_B!(ar, pirfft, ah)
    a .= convert.(Complex{Float64}, ar)
  end
end






# Initialize random number generator
srand(123)
effort = FFTW.MEASURE
nloops = 400
nthreads = Sys.CPU_CORES
FFTW.set_num_threads(nthreads)

for N in [32, 64, 128, 256, 512, 1024]

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
  fftloop(a, ah, pfft, pifft, 1)
  realfftloop(ar, arh, prfft, pirfft, 1)
  realfftloop(a, ar, arh, prfft, pirfft, 1)

  @printf "N: %5d^2, threads: %d, %24s : " n nthreads "complex fft"
  @time fftloop(a, ah, pfft, pifft, nloops)

  @printf "N: %5d^2, threads: %d, %24s : " n nthreads "real fft"
  @time realfftloop(ar, arh, prfft, pirfft, nloops)

  @printf "N: %5d^2, threads: %d, %24s : " n nthreads "real w reassign fft"
  @time realfftloop(a, ar, arh, prfft, pirfft, nloops)

  println()

end
