#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#pragma once

template <typename Algebra>
struct NormalDistribution {
  using Scalar = typename Algebra::Scalar;
  using VectorX = typename Algebra::VectorX;
  VectorX mu;
  VectorX sigma;
  VectorX sqsigma;
  NormalDistribution(const VectorX &mu, const VectorX &sigma)
      : mu(mu),
        sigma(sigma),
        sqsigma((sigma.array() * sigma.array()).matrix()) {}

  VectorX CDF(const VectorX &x) const {
    VectorX ret(1);
    ret << (1.0 + std::erf(M_SQRT1_2 * (x[0] - mu[0]) / sigma[0])) / 2.0;
    return ret;
  }

  Scalar LogPDF(const VectorX &x) const {
    Scalar logpdf =
        -Algebra::from_double(0.5) * Algebra::log(Algebra::norm(sigma));
    logpdf -= (x - mu).dot(((x - mu).array() / sigma.array()).matrix());
    logpdf -= Algebra::size(x) *
              Algebra::log(Algebra::pi() * Algebra::from_double(2.));
    return logpdf;
  }

  VectorX DLogPDF(const VectorX &x) const {
    return Algebra::cdiv(mu - x, sqsigma);
  }
  auto DLogPDF() const {
    return [this](const VectorX &x) { return DLogPDF(x); };
  }
};

#endif  // DISTRIBUTIONS_H