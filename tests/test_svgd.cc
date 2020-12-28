#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "SVGD.h"
#include "distributions.h"
#include "eigen_algebra.h"

/**
 * Computes the CDF of the Kolmogorov-Smirnov distribution.
 *
 * @param z Value
 * @returns KS CDF at z
 */
double KolmogorovSmirnovCDF(double z) {
  if (z == 0.0) {
    return 0.0;
  }
  const double sqz = z * z;
  if (z < 1.18) {
    double P = 0.0;
    for (int j = 1; j < 4; ++j) {
      int j2m1pi = M_PI * (2 * j - 1);
      P += std::exp(-(j2m1pi * j2m1pi) / (8 * sqz));
    }
    return std::sqrt(M_PI * 2) * P / z;
  } else {
    double P = 0.0;
    for (int j = 1; j < 4; ++j) {
      P += ((j - 1) % 2 ? -1 : 1) * std::exp(-2 * j * j * sqz);
    }
    return 1 - 2 * P;
  }
}

/**
 * Computes the Komogorov-Smirnov test statistic for given particles and a
 * continuous distirbution.
 *
 * @param x Particles sampled from a distribution.
 * @param dist Target distribution
 * @returns p-value computed from the K-S distance between the
 * distributions.
 */
template <typename Particles, typename Distribution>
double KolmogorovSmirnovTest(const Particles& x, const Distribution& dist) {
  Particles data = x;  // Copy
  std::sort(data.begin(), data.end(),
            [](const auto& a, const auto& b) { return a[0] < b[0]; });
  double max_distance = 0.0;
  double n = data.size();
  for (int i = 0; i < data.size(); ++i) {
    const double ecdf0 = i / n;
    const double ecdf1 = (i + 1.0) / n;
    const double cdf = dist.CDF(data[i])[0];
    max_distance = std::max(max_distance, std::fabs(cdf - ecdf0));
    max_distance = std::max(max_distance, std::fabs(cdf - ecdf1));
  }
  const double rt_n = std::sqrt(data.size());
  return 1 - KolmogorovSmirnovCDF((rt_n + 0.12 + 0.11 / rt_n) * max_distance);
}

/**
 * A reference implementation that more closely matches the original math in
 * the paper.
 */
template <typename Algebra, template <typename> typename Kernel,
          typename Particles, typename DLnProb>
void ReferenceStep(const typename Algebra::Scalar& step_size, DLnProb dlnprob,
                   const Particles& x0, Particles* x1) {
  Kernel<Algebra> k(Algebra::one());
  for (int i = 0; i < x0.size(); ++i) {
    // Compute phi-star.
    (*x1)[i] = Algebra::zerox(x0[i].size());
    for (int j = 0; j < x0.size(); ++j) {
      const auto& gradlnp = dlnprob(x0[j]);
      const auto& [kxjxi, gradkxjxi] = k.D01(x0[j], x0[i]);
      (*x1)[i] += kxjxi * gradlnp + gradkxjxi;
    }
    (*x1)[i] *= step_size / x0.size();
    (*x1)[i] += x0[i];
  }
}

TEST(SVGD, KolmogorovSmirnov) {
  constexpr int n = 100;

  std::vector<Eigen::VectorXd> x, y;
  for (int i = 0; i < n; ++i) {
    x.push_back(Eigen::VectorXd::Random(1));
  }
  y.resize(x.size());

  auto mu = Eigen::VectorXd::Constant(1, 5.0);
  auto sigma = Eigen::VectorXd::Constant(1, 1.0);
  NormalDistribution<EigenAlgebra> dist(mu, sigma);
  auto dlnprob = dist.DLogPDF();

  for (int i = 0; i < 1000; ++i) {
    svgd::Step<EigenAlgebra, svgd::kernels::RBF>(0.1, dlnprob, x, &y);
    std::swap(x, y);
  }

  const double p = KolmogorovSmirnovTest(x, dist);
  ASSERT_GE(p, 0.95);
}

TEST(SVGD, ReferenceStep) {
  constexpr int n = 3;

  std::vector<Eigen::VectorXd> x, y, y_ref;
  for (int i = 0; i < n; ++i) {
    x.push_back(Eigen::VectorXd::Random(1));
  }
  y.resize(x.size());
  y_ref.resize(x.size());

  auto mu = Eigen::VectorXd::Constant(1, 5.0);
  auto sigma = Eigen::VectorXd::Constant(1, 1.0);
  NormalDistribution<EigenAlgebra> dist(mu, sigma);
  auto dlnprob = dist.DLogPDF();

  for (int i = 0; i < 1000; ++i) {
    svgd::Step<EigenAlgebra, svgd::kernels::RBF>(0.1, dlnprob, x, &y);
    ReferenceStep<EigenAlgebra, svgd::kernels::RBF>(0.1, dlnprob, x, &y_ref);
    y[0][0] += 1;
    for (int j = 0; j < y.size(); ++j) {
      ASSERT_FLOAT_EQ(y[j][0], y_ref[j][0]) << "Index: " << j;
    }
    std::swap(x, y);
  }

  const double p = KolmogorovSmirnovTest(x, dist);
  ASSERT_GE(p, 0.95);
}