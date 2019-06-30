# Numerical Optimisation

Numerical Optimisation is a C++ library for executing and testing different unconstrained optimisation algorithms. The project is based on Vilin written in MATLAB. Although this library is standalone, the idea is to eventually delegate heavy Vilin computation directly to C++ to speed it up.

## Vilin, a MATLAB GUI application for Numerical optimisation

https://github.com/markomil/vilin-numerical-optimization


## Project Structure
```
.
├── CMakeLists.txt
├── functions/
│   ├── almost_pert_quad.h
│   ├── cube.h
│   ├── diagonal1.h
│   ├── explin1.h
│   ├── ext_himmelblau.h
│   ├── ext_psc1.h
│   ├── ext_quad_pen_qp1.h
│   ├── ext_quad_pen_qp2.h
│   ├── ext_rosenbrock.h
│   ├── fletchcr.h
│   ├── full_hessian2.h
│   ├── function.h
│   ├── gen_psc1.h
│   ├── gen_rosenbrock.h
│   ├── part_pert_quad.h
│   └── raydan1.h
├── functions.h
├── library.cpp
├── library.h
├── line_searches/
│   ├── armijo.h
│   ├── base_line_search.h
│   ├── binary.h
│   ├── fixed_step_size.h
│   ├── goldstein.h
│   ├── strong_wolfe.h
│   └── wolfe.h
├── line_searches.h
├── main.cpp
├── methods/
│   ├── base_method.h
│   ├── conjugate_gradient/
│   │   └── fletcher_reeves.h
│   ├── gradient_descent/
│   │   ├── gradient_descent.h
│   │   └── momentum.h
│   └── quasi_newton/
│       ├── bfgs.h
│       ├── dfp.h
│       ├── l_bfgs.h
│       └── sr1.h
├── methods.h
├── README.md
└── utilities/
    └── linear_algebra.h

```

## Authors

* **Ivan Stosic** - ivan100sic@gmail.com
* **Igor Stosic** - igor.stosic@pmf.edu.rs
* **Lazar Stamenkovic** - lazar.stamenkovic@pmf.edu.rs

