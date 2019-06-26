# C++ Backend for Vilin

C++ library for executing and testing different unconstrained optimization algorithms. 

##Vilin

#### Matlab GUI application for Numerical optimization

https://github.com/markomil/vilin-numerical-optimization.git


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
├── linear_algebra.h
├── line_searches/
│   ├── armijo.h
│   ├── base_line_search.h
│   ├── binary.h
│   ├── fixed_line_search.h
│   ├── goldstein.h
│   ├── strong_wolfe.h
│   └── wolfe.h
├── line_searches.h
├── main.cpp
├── methods/
│   ├── base_method.h
│   ├── conjugate_gradient/
│   ├── gradient_descent/
│   └── quasi_newton/
├── methods.h
└── README.md

```

## Built With

* [C++ STL](https://github.com/cjlin1/liblinear) - The Standard Template Library (STL) is a set of C++ template classes to provide common programming data structures and functions such as lists, stacks, arrays, etc. It is a library of container classes, algorithms, and iterators.

## Authors

* **Lazar Stamenkovic** - lazar.stamenkovic@pmf.edu.rs
* **Igor Stosic** - igor.stosic@pmf.edu.rs
* **Ivan Stosic** - ivan100sic@gmail.com

See also the list of [contributors](https://github.com/lazarst96/NumericalOptimisation) who participated in this project.