#ifndef E78F_SVGD_H
#define E78F_SVGD_H

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

/* Copyright 2020 David Millard <dmillard10@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Header-only implementation of the Stein Variational gradient descent
 * algorithm for Bayesian optimization, presented in [1].
 *
 * [1] Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General
 *     Purpose Bayesian Inference Algorithm,” 2016.
 */

namespace svgd {

namespace kernels {
template <typename Algebra>
struct RBF {
  using Scalar = typename Algebra::Scalar;
  using VectorX = typename Algebra::VectorX;

  Scalar h;

  /**
   * Creates an RBF kernel with bandwidth h
   *
   * @param h Bandwidth of the RBF
   */
  RBF(const Scalar &h) : h(h) {}

  /**
   * Computes an RBF similarity kernel between x and y
   *
   * @param x First input point
   * @param y Second input point
   * @returns RBF kernel similarity between x and y
   */
  Scalar operator()(const VectorX &x, const VectorX &y) {
    const auto &sqnorm = Algebra::sqnorm(x - y);
    return Algebra::exp(-sqnorm / (h * h * Algebra::two()));
  }

  /**
   * Computes the RBF kernel of x and y and the derivative w.r.t x.
   *
   * @param x First input point
   * @param y Second input point
   * @returns A pair {k(x, y), dk/dx}
   */
  std::pair<Scalar, VectorX> D01(const VectorX &x, const VectorX &y) {
    Scalar k = (*this)(x, y);
    return {k, -k * (x - y) / (h * h)};
  }
};
}  // namespace kernels

template <typename Algebra,
          typename Particles = std::vector<typename Algebra::VectorX>>
struct SVGD {
  using Scalar = typename Algebra::Scalar;
  using VectorX = typename Algebra::VectorX;

  Scalar bandwidth{Algebra::from_double(-1)};
  Scalar step_size{Algebra::from_double(5e-3)};

  /**
   * Adaptive learning rate updated by Adam (not to be defined by the user).
   */
  Scalar learning_rate{Algebra::zero()};

  Scalar beta1{Algebra::from_double(0.9)};
  Scalar beta2{Algebra::from_double(0.999)};

  Scalar epsilon{Algebra::from_double(1e-8)};

  int step{0};

  std::vector<std::pair<Scalar, Scalar>> parameter_bounds;

 private:
  // momentum
  Particles m_;
  // accumulates past gradient variance
  Particles v_;

 public:
  /**
   * Takes a single step of SVGD
   *
   * @param dlnprob Derivative of the log pdf of the target distribution
   * @param x0 Initial particles
   * @param x1 Particles after step
   */
  template <template <typename> typename Kernel, typename DLnProb>
  void Step(DLnProb dlnprob, const Particles &x0, Particles *x1) {
    using Scalar = typename Algebra::Scalar;
    const Scalar one = Algebra::one();

    if (m_.size() != x0.size()) {
      // initialize
      m_ = std::vector<VectorX>(x0.size(), Algebra::zerox(x0[0].size()));
    }
    if (v_.size() != x0.size()) {
      // initialize
      v_ = std::vector<VectorX>(x0.size(), Algebra::zerox(x0[0].size()));
    }
    if (bandwidth < 0.) {
      // Use pairwise particle distance median as heuristic.
      // Only compute the unique pairwise squared distances.
      std::vector<Scalar> dists(x0.size() * (x0.size() - 1) / 2);
      std::size_t di = 0;
      for (std::size_t i = 0; i < x0.size(); ++i) {
        for (std::size_t j = i + 1; j < x0.size(); ++j, ++di) {
          dists[di] = Algebra::dot(x0[i], x0[j]);
        }
      }
      std::sort(dists.begin(), dists.end());
      bandwidth = std::sqrt(Scalar(0.5) * dists[dists.size() / 2] /
                            Algebra::log((double)x0.size() + one));
    }
    ++step;
    Kernel<Algebra> kernel(bandwidth);
    const int n = static_cast<int>(x0.size());
    for (int i = 0; i < n; ++i) {
      (*x1)[i] = Algebra::zerox(x0[i].size());
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
      // Compute phi-star
      for (int j = 0; j < n; ++j) {
        const auto &gradlnp = dlnprob(x0[j]);
        assert(!gradlnp.hasNaN());
        const auto &[kxjxi, gradkxjxi] = kernel.D01(x0[j], x0[i]);
        (*x1)[i] += kxjxi * gradlnp + gradkxjxi;
      }
    }

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      (*x1)[i] /= Algebra::from_double((double)n);
      // Adam update
      const auto &grad = (*x1)[i];
      m_[i] = beta1 * m_[i] + (one - beta1) * grad;
      v_[i] = beta2 * v_[i] + (one - beta1) * grad.array().square().matrix();
      Scalar t = Algebra::from_double((double)step);
      Scalar bias_correction1 = one / (one - Algebra::pow(beta1, t));
      Scalar bias_correction2 = one / (one - Algebra::pow(beta2, t));
      learning_rate =
          step_size * bias_correction1 / Algebra::sqrt(bias_correction2);
      // (*x1)[i] = x0[i].array() + step_size * mt.array() / (vt.array().sqrt()
      // + epsilon);
      for (int j = 0; j < x0[i].size(); ++j) {
        Scalar delta =
            learning_rate * m_[i][j] / (Algebra::sqrt(v_[i][j]) + epsilon);
        (*x1)[i][j] = x0[i][j] + delta;
        if (j < parameter_bounds.size()) {
          double wouldbe = Algebra::to_double((*x1)[i][j]);
          // flip derivative if particle dimension would be out of bounds
          if (wouldbe < parameter_bounds[j].first ||
              wouldbe > parameter_bounds[j].second) {
            (*x1)[i][j] = x0[i][j] - delta;
          }
        }
      }
    }
  }
};  // namespace svgd

}  // namespace svgd

#endif