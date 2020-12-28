# SteinVariationalGradientDescent

Header-only implementation of the Stein Variational gradient descent algorithm for Bayesian optimization, presented in \[1\]. 

\[1\] Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,” 2016.

## Build

Requires Eigen.

To build tests and benchmarks:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build build
```