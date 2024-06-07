# IsasJuliaApi
A coherent API, giving access to sampling methods developed at ISAS.

Served via the Cloudrunner https://github.com/KIT-ISAS/Cloudrunner

## Usage
```
import Pkg; Pkg.add('https://github.com/KIT-ISAS/IsasJuliaApi')
using IsasJuliaApi
X = sample_LCD_Gauss_LCDHQ(; C=[0.5 0; 0 1], L=10)
```
