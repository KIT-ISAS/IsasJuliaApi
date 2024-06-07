module IsasJuliaApi

using LinearAlgebra: eigen

export sample_LCD_Gauss_LCDHQ

# import Pkg; Pkg.add.(["Revise","QuadGK","ForwardDiff","Optimization","OptimizationOptimJL"])
# using Revise
# ] dev .
# ] develop .
# using IsasJuliaApi
# sample_LCD_Gauss_LCDHQ()

greet() = print("Hello World!")

include("LCDGauss.jl")

"2D anisotropic Gaussian high quality LCD samples. 
- C: [0.5 0; 0 1] Covariance matrix
- L: 10, Number of Samples"
function sample_LCD_Gauss_LCDHQ(; C=[0.5 0; 0 1], L=10)
    F = eigen(C)
    x = LCDGauss.sample(sqrt.(F.values); L) 
    x = F.vectors * x
end

end # module IsasJuliaApi

