"LCD Gaussian Samples
Reference: Hanebeck, Huber, Klumpp, 'Dirac Mixture Approximation of Multivariate Gaussian Densities', CDC 2009,
with some changes of formula in Theorem III.2 according to Matlab code from Uwe Hanebeck:
svn checkout svn+ssh://i81server.iar.kit.edu/SVN/Publ/2009/DM_Gauss_Approx 
"
module LCDGauss

using QuadGK, ForwardDiff
using Optimization
import OptimizationOptimJL
# import OptimizationMOI, Ipopt

export D, sample

# Notation
#  - N : dimension 
#  - L : number of samples 
#  - x : samples 
#  - b : kernel width 
#  - σ : standard deviation

"Draw Gaussian samples. Anisotropic (non-standard normal) but axis-aligned (independent)
Inputs
    - σ    : vector N-elm  : standard deviations along dimensions
    - L    : scalar        : number of samples 
    - bmin : scalar
    - bmax : scalar
    - solver 
    - solver_reltol : scalar 
    - quad_reltol   : scalar 
    - fast          : scalar bool

Review Solvers                                      L=20   L=50
    - OptimizationOptimJL.ConjugateGradient()         1      6 
    - OptimizationOptimJL.GradientDescent()          20     40
    - OptimizationOptimJL.BFGS()                      4     14
    - OptimizationOptimJL.LBFGS()                     3     23
    - OptimizationOptimJL.NewtonTrustRegion()         6     56
    - OptimizationOptimJL.Newton()                   23      - 
    - Optim.KrylovTrustRegion()                      60      - 
    - Ipopt.Optimizer()                               8     75

Review bmin, bmax 
    - bmin = 0.1; bmax = 10       # Steinbring C++, for standard normal, N<10
    - bmin = 0.0001; bmax = 100   # UDH Matlab, 2D anisotropic Gauss
"
function sample(σ::Vector{<:Number}; L=5::Integer, bmin=0.0001::Real, bmax=100::Real, solver=OptimizationOptimJL.ConjugateGradient(), solver_reltol::Real=1e-6, quad_reltol::Real=1e-10, fast::Bool=true) 
    @assert L≥0 
    @assert bmax>0 
    @assert all(σ.>0)
    N = length(σ)
    @assert N==2
    # initial guess: random 
    x0 = randn(N,L) .* σ 
    w = ones(L)/L
    D1 = Di(;x=x0,w,σ,bmin,bmax,quad_reltol,PFunc=P1) # pre-calculate the constant term D1 
    fun = (x,p) -> D(;x=reshape(x,(N,L)),w,σ,bmin,bmax,quad_reltol,D1,fast)
    fun(vec(x0),0) # test run
    optfun = OptimizationFunction(fun, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optfun, vec(x0))

    sol = solve(prob, solver, reltol=solver_reltol) 
    # display(sol.original)

    xOpt = reshape(sol.u,(N,L))
    return xOpt
end

# Literal Implementation of Theorem III.1 

"Complete distance function
Inputs
  - x : matrix NxL
  - w : vector L-elm
  - b : scalar
  - σ : vector N-elm
"
function D(; x, w, σ, bmin, bmax, quad_reltol, D1, fast)
    if fast
        # fast version
        D1 = D1 # D1 is constant, compute only once at beginning
        D3 = D3_approx(; x, w, bmax)
    else
        # slow version
        D1 = Di(;x,w,σ,bmin,bmax,quad_reltol,PFunc=P1) 
        D3 = Di(;x,w,σ,bmin,bmax,quad_reltol,PFunc=P3) 
    end
    D2 = Di(;x,w,σ,bmin,bmax,quad_reltol,PFunc=P2)
    return D1 - 2D2 + D3
end

"Weighting function"
function wb(; b, bmax, N)
    if 0 ≤ b ≤ bmax
        1/(b^(N-1))
    else
        0
    end
end

"Di, integral of Pi"
function Di(; x, w, σ, bmin, bmax, quad_reltol, PFunc)
    N = size(x,1)
    fun = b -> wb(;b,bmax,N) * PFunc(;x,w,σ,b)
    fun(1) # test run
    integral, err = quadgk(fun, bmin, bmax, rtol=quad_reltol, atol=0) 
    return integral
end

"Inner function P1, to be integrated.  
Inputs
  - b : scalar
  - σ : vector N-elm
"
function P1(; x, w, σ, b)
    N = size(x,1)
    return π^(N/2) * b^(2N) * prod(1 ./ sqrt.(σ.^2 .+ b^2)) 
end

"Inner function P2, to be integrated. 
Inputs
  - x : matrix NxL
  - w : vector L-elm
  - b : scalar
  - σ : vector N-elm
"
function P2(; x, w, σ, b)
    (N,L) = size(x)
    return (2π)^(N/2) * b^(2N) * prod(1 ./ sqrt.(σ.^2 .+ 2b^2)) * sum(w'.*exp.(-1/2 * sum(dims=1, x.^2 ./ (σ.^2 .+ 2b^2))))
end

"Inner function P3, to be integrated. 
Inputs
  - x : matrix NxL
  - w : vector L-elm
  - b : scalar
  - σ : vector N-elm
"
function P3(; x, w, σ, b)
    (N,L) = size(x)
    r = 0.
    for i=1:L
        for j=1:L
            r += w[i] * w[j] * exp(-1/2 * sum((x[:,i].-x[:,j]).^2) / (2b^2))
        end
    end
    return r * π^(N/2) * b^N 
end

"Closed form approximation of D3. 
Inputs
  - x : matrix NxL
  - w : vector L-elm
  - b : scalar
"
function D3_approx(; x, w, bmax)
    (N,L) = size(x)
    Cb = log(4*bmax^2) - MathConstants.eulergamma
    r = 0.
    for i=1:L
        for j=1:L
            Tij = sum((x[:,i].-x[:,j]).^2)
            #r += π^(N/2)/8 * w[i] * w[j] * (4*bmax^2 - Cb*Tij + xlog(Tij)) # Paper (doesn't work!)
            r += π/8 * w[i] * w[j] * (4*bmax^2*exp(-1/2*Tij/(2*bmax^2)) - Cb*Tij + xlog(Tij) - Tij^2/(4*bmax^2)) # UDH Matlab code 
        end
    end
    return r 
end

function xlog(z)
    if z==0
        return 0
    else
        return z * log(z)
    end
end

end #module
