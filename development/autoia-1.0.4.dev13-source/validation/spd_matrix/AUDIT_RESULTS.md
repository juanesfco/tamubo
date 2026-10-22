# SPD matrix-family high-precision audit

The audit uses 80-decimal-digit `mpmath` calculations at a 9 x 9 grid
inside each of four nested hyperparameter boxes for a 3 x 3 RBF-like SPD
matrix family. Sampling is only an independent challenge; AutoIA does not
use these samples to construct its certificates.

- total high-precision scalar containment checks: **972**
- all contained: **True**
- wall time: **9.099 s**

| log-box halfwidth | AutoIA widths (logdet, q, combined) | sampled widths | contained |
|---:|---|---|:---:|
| 0.12 | `[0.29362033 0.19771369 0.05155613]` | `[0.2907338 0.1941732 0.0482803]` | True |
| 0.06 | `[0.1444301  0.09541763 0.02458551]` | `[0.14436516 0.09533909 0.02451304]` | True |
| 0.03 | `[0.07205783 0.04745431 0.01230391]` | `[0.07205605 0.04745218 0.01230193]` | True |
| 0.015 | `[0.03601222 0.02369902 0.00615666]` | `[0.03601217 0.02369895 0.00615661]` | True |

The certified widths decrease as the physical hyperparameter box shrinks.
