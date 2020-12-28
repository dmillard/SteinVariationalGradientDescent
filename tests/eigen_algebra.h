#ifndef EIGEN_ALGEBRA_H
#define EIGEN_ALGEBRA_H

#pragma once

#include <Eigen/Dense>

struct EigenAlgebra {
  using Scalar = double;
  using VectorX = Eigen::VectorXd;

  static Scalar zero() { return 0.; }
  static Scalar one() { return 1.; }
  static Scalar two() { return 2.; }

  static VectorX zerox(int dim) { return VectorX::Zero(dim); }

  static Scalar exp(const Scalar& x) { return std::exp(x); }
  static VectorX cdiv(const VectorX& x, const VectorX& y) {
    return x.array() / y.array();
  }
  static VectorX csquare(const VectorX& x) { return x.array().square(); }
  static Scalar sqnorm(const VectorX& x) { return x.squaredNorm(); }
};

#endif  // EIGEN_ALGEBRA_H