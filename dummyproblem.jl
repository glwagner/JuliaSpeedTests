module DummyProblem

export Grid, Vars, init_grid, init_vars, step_nsteps!

mutable struct Grid
  nx::Int

  nk::Int
  nl::Int

  k::Array{Float64, 1}
  l::Array{Float64, 1}

  ik::Array{Complex128, 1}
  il::Array{Complex128, 1}

  invksq::Array{Float64, 2}

  # FFT plans!
  fftplan::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,false,2}

  ifftplan::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,false,2},Float64}

  rfftplan::Base.DFT.FFTW.rFFTWPlan{Float64,-1,false,2}

  irfftplan::Base.DFT.ScaledPlan{Complex{Float64},
    Base.DFT.FFTW.rFFTWPlan{Complex{Float64},1,false,2},Float64}
  
end
  

mutable struct Vars
  a::Array{Float64, 2}
  b::Array{Float64, 2}
  c::Array{Float64, 2}

  ah::Array{Complex128, 2}
  bh::Array{Complex128, 2}
  ch::Array{Complex128, 2}
end



function init_grid(nx::Int)

  nk = Int(nx/2+1)
  nl = nx

  k = Array{Float64}(nk)
  l = Array{Float64}(nl)

  ik = Array{Complex128}(nk)
  il = Array{Complex128}(nl)

  invksq = Array{Float64}(nk, nl)

  i1 = 0:1:nx/2
  i2 = (-nx/2+1):1:-1

  k = i1
  l = cat(1, i1, i2)

  ik = im*k
  il = im*l

  for j = 1:nl
    for i = 1:nk
      if i == 1 && j == 1 
        invksq[i, j] = 0.0
      else
        invksq[i, j] = 1.0/(k[i]^2.0 + l[j]^2.0)
      end
    end
  end

  # FFT plans; use grid vars.
  effort = FFTW.MEASURE

  fftplan   =   plan_fft(Array{Float64,2}(nx, nx); flags=effort)
  ifftplan  =  plan_ifft(Array{Complex128,2}(nk, nl); flags=effort)

  rfftplan  =  plan_rfft(Array{Float64,2}(nx, nx); flags=effort)
  irfftplan = plan_irfft(Array{Complex128,2}(nk, nl), nx; flags=effort)

  Grid(nx, nk, nl, k, l, ik, il, invksq, 
    fftplan, ifftplan, rfftplan, irfftplan)

end


function init_vars(g::Grid)
  a = Array{Float64}(g.nx, g.nx)
  b = Array{Float64}(g.nx, g.nx)
  c = Array{Float64}(g.nx, g.nx)

  ah = Array{Complex128}(g.nk, g.nl)
  bh = Array{Complex128}(g.nk, g.nl)
  ch = Array{Complex128}(g.nk, g.nl)

  srand(123)
  a = rand(g.nx, g.nx)
  b = rand(g.nx, g.nx)
  c = rand(g.nx, g.nx)

  ah = rfft(a)
  bh = rfft(b)
  ch = rfft(c)

  return Vars(a, b, c, ah, bh, ch)

end


#function step_nsteps!(nsteps::Int, v::Vars, g::Grid)
#
#  for step = 1:nsteps
#
#    for j = 1:g.nx
#      for i = 1:g.nx
#        v.c[i, j] = v.a[i, j]*v.b[i, j]
#      end
#    end
#
#    v.ch = rfft(v.c)
#
#    for j = 1:g.nl
#      for i = 1:g.nk
#        @inbounds v.ah[i, j] = g.ik[i]*v.ch[i, j]*g.invksq[i, j]
#        @inbounds v.bh[i, j] = g.il[j]*v.ch[i, j]*g.invksq[i, j]
#      end
#    end
#
#    v.a = irfft(v.ah, g.nx)
#    v.b = irfft(v.bh, g.nx)
#
#  end
#
#end



function step_nsteps!(nsteps::Int, v::Vars, g::Grid)

  for step = 1:nsteps

    v.c .= v.a .* v.b

    A_mul_B!(v.ch, g.rfftplan, v.c)

    for j = 1:g.nl
      for i = 1:g.nk
        @inbounds v.ah[i, j] = g.ik[i]*v.ch[i, j]*g.invksq[i, j]
        @inbounds v.bh[i, j] = g.il[j]*v.ch[i, j]*g.invksq[i, j]
      end
    end

    A_mul_B!(v.a, g.irfftplan, v.ah)
    A_mul_B!(v.b, g.irfftplan, v.bh)

  end

end


end
