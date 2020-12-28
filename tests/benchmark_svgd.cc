#include <benchmark/benchmark.h>

#include <functional>
#include <random>

#include "SVGD.h"
#include "distributions.h"
#include "eigen_algebra.h"

static void BM_SVGDStep(benchmark::State& state) {
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
  for (auto _ : state) {
    svgd::Step<EigenAlgebra, svgd::kernels::RBF>(1e-3, dlnprob, x, &y);
  }
}
BENCHMARK(BM_SVGDStep);

// Run the benchmark
BENCHMARK_MAIN();