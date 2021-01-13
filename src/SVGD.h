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
  RBF(const Scalar& h) : h(h) {}

  /**
   * Computes an RBF similarity kernel between x and y
   *
   * @param x First input point
   * @param y Second input point
   * @returns RBF kernel similarity between x and y
   */
  Scalar operator()(const VectorX& x, const VectorX& y) {
    const auto& sqnorm = Algebra::sqnorm(x - y);
    return Algebra::exp(-sqnorm / (h * h * Algebra::two()));
  }

  /**
   * Computes the RBF kernel of x and y and the derivative w.r.t x.
   *
   * @param x First input point
   * @param y Second input point
   * @returns A pair {k(x, y), dk/dx}
   */
  std::pair<Scalar, VectorX> D01(const VectorX& x, const VectorX& y) {
    Scalar k = (*this)(x, y);
    return {k, -k * (x - y) / (h * h)};
  }
};
}  // namespace kernels

/**
 * Takes a single step of SVGD
 *
 * @param step_size Size of the step
 * @param dlnprob Derivative of the log pdf of the target distribution
 * @param x0 Initial particles
 * @param x1 Particles after step
 */
template <typename Algebra, template <typename> typename Kernel,
          typename Particles, typename DLnProb>
void Step(const typename Algebra::Scalar& step_size, DLnProb dlnprob,
          const Particles& x0, Particles* x1,
          typename Algebra::Scalar bandwidth = -1) {
  using Scalar = typename Algebra::Scalar;
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
                          Algebra::log((double)x0.size() + Algebra::one()));
  }
  Kernel<Algebra> kernel(bandwidth);
  for (int i = 0; i < x0.size(); ++i) {
    // Compute phi-star.
    (*x1)[i] = Algebra::zerox(x0[i].size());
    for (std::size_t j = 0; j < x0.size(); ++j) {
      const auto& gradlnp = dlnprob(x0[j]);
      const auto& [kxjxi, gradkxjxi] = kernel.D01(x0[j], x0[i]);
      (*x1)[i] += kxjxi * gradlnp + gradkxjxi;
    }
    (*x1)[i] *= step_size / x0.size();
    (*x1)[i] += x0[i];
  }
}

}  // namespace svgd

#endif