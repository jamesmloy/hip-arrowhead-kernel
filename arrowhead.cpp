#include "hip/hip_runtime.h"
#include "deps/json.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <fstream>

#include "utils.hpp"

using json = nlohmann::json;

std::random_device rd;
std::mt19937 gen(rd());

__global__ void
xlCalculation(int const valsPerBlock,
              double const* botRows,
              double const* sources,
              double const* diags,
              double const* rightCols,
              double const* cornerVals,
              double const* lastSource,
              double* xl);

__global__ void
xiCalculation(int const valsPerCell,
              int const totVals,
              double const* xl,
              double const* rightCols,
              double const* diags,
              double const* sources,
              double* xi);

using VecOfVec = ContiguousVecOfVec<double>;

json getConfig(std::string const& configFile)
{
  try
  {
    std::cout << "Reading config file\n";
    std::ifstream f(configFile);
    return json::parse(f);
  }
  catch (std::exception const& e)
  {
    std::cout << "Could not read config file " << configFile
              << " because of exception:\n"
              << e.what() << std::endl;
    return 1;
  }
}

int
main()
{
  std::string const configFile = "config.json";
  std::string const resultFile = "results.json";

  auto const config = getConfig(configFile);

  int const nCells = config["cell_counts"][0];
  int const nKcells = config["kcell_counts"][0];
  int const deviceId = 0;
  int const threadsPerBlock = 128;
  int const numRows = nCells * nKcells;
  int const blocks = (numRows + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blockDim{ static_cast<uint32_t>(blocks), 1, 1 };
  dim3 threadDim{ static_cast<uint32_t>(threadsPerBlock), 1, 1 };
  gen.seed(1);

  hipSetDevice(deviceId);
  auto const props = getDeviceProps(deviceId);
  std::cout << props << std::endl;

  // generate host arrays

  // diagonal
  auto diags = VecOfVec(nCells, nKcells);
  diags.randomize(gen, 0.5, 1.5);

  // right columns, set these as opposite the
  // diagonal
  auto rightCols = VecOfVec(nCells, nKcells);
  for (int i = 0; i < nCells * nKcells; ++i)
    rightCols[i] = -diags[i] * 0.5;

  // bottom rows
  auto botRows = VecOfVec(nCells, nKcells);
  botRows.randomize(gen, 0.5, 1.5);

  // the corner value of the matrix is
  // the opposite sum of the bottom row
  std::vector<double> cornerVals;
  cornerVals.reserve(nCells);
  for (int i = 0; i < nCells; ++i) {
    auto begEnd = botRows.row(i);
    cornerVals.push_back(-std::accumulate(begEnd.first, begEnd.second, 0.0));
  }

  // xl value, this will store the first result
  std::vector<double> xlValues(nCells, 0);

  // expected xl result
  std::vector<double> xlExpected(nCells, 0);
  std::uniform_real_distribution<double> dis(0.5, 1.5);
  for (int i = 0; i < nCells; ++i)
    xlExpected[i] = dis(gen);

  // xi value, this will store the second result
  auto xiValues = VecOfVec(nCells, nKcells);

  // expected xi result
  auto xiExpected = VecOfVec(nCells, nKcells);
  xiExpected.randomize(gen, 0.5, 1.5);

  // the last source term
  std::vector<double> lastSource(nCells, 0);

  // sources
  auto source = VecOfVec(nCells, nKcells);

  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nKcells; ++k) {
      int const i = c * nKcells + k;
      source[i] = diags[i] * xiExpected[i] + rightCols[i] * xlExpected[c];
      lastSource[c] += botRows[i] * xiExpected[i];
    }
    lastSource[c] += cornerVals[c] * xlExpected[c];
  }

#if DIAGNOSTIC
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nKcells; ++k) {
      int const i = c * nKcells + k;
      std::cout << "diag: " << diags[i] << ", right: " << rightCols[i]
                << ", bot: " << botRows[i] << ", source: " << source[i]
                << ", xi: " << xiExpected[i] << "\n";
    }
    std::cout << "corner: " << cornerVals[c]
              << ", last source: " << lastSource[c] << ", xl: " << xlExpected[c]
              << "\n";
  }
#endif

  // generate device arrays
  auto const numBytes = diags.numBytes();

  // diagonals
  double* devDiag = nullptr;
  // right columns
  double* devRightCols = nullptr;
  // bottom rows
  double* devBotRows = nullptr;
  // source term
  double* devSource = nullptr;
  // result vector for the second result
  double* devXi = nullptr;
  // corner value of the matrix
  double* devCornerVals = nullptr;
  // the last source term
  double* devLastSources = nullptr;
  // result vector for the first result
  double* devXl = nullptr;

  {
    // diagonals
    HIP_CHECK(hipMalloc(&devDiag, numBytes));
    HIP_CHECK(
      hipMemcpy(devDiag, diags.data(), numBytes, hipMemcpyHostToDevice));

    // right columns
    HIP_CHECK(hipMalloc(&devRightCols, numBytes));
    HIP_CHECK(hipMemcpy(
      devRightCols, rightCols.data(), numBytes, hipMemcpyHostToDevice));

    // bottom rows
    HIP_CHECK(hipMalloc(&devBotRows, numBytes));
    HIP_CHECK(
      hipMemcpy(devBotRows, botRows.data(), numBytes, hipMemcpyHostToDevice));

    // source term
    HIP_CHECK(hipMalloc(&devSource, numBytes));
    HIP_CHECK(
      hipMemcpy(devSource, source.data(), numBytes, hipMemcpyHostToDevice));

    // result vector for the second result
    HIP_CHECK(hipMalloc(&devXi, numBytes));
    HIP_CHECK(hipMemcpy(devXi, source.data(), numBytes, hipMemcpyHostToDevice));

    // corner value of the matrix
    HIP_CHECK(hipMalloc(&devCornerVals, nCells * sizeof(double)));
    HIP_CHECK(hipMemcpy(devCornerVals,
                        cornerVals.data(),
                        nCells * sizeof(double),
                        hipMemcpyHostToDevice));

    // the last source term
    HIP_CHECK(hipMalloc(&devLastSources, nCells * sizeof(double)));
    HIP_CHECK(hipMemcpy(devLastSources,
                        lastSource.data(),
                        nCells * sizeof(double),
                        hipMemcpyHostToDevice));

    // result vector for the first result
    HIP_CHECK(hipMalloc(&devXl, nCells * sizeof(double)));
    HIP_CHECK(hipMemcpy(devXl,
                        lastSource.data(),
                        nCells * sizeof(double),
                        hipMemcpyHostToDevice));
  }

  std::cout << "\n============================================================="
               "=========\n\n";

  {
    dim3 reduceBlockDim{ static_cast<uint32_t>(nCells), 1, 1 };
    dim3 reduceThreadDim{ static_cast<uint32_t>(threadsPerBlock), 1, 1 };
    hipLaunchKernelGGL(xlCalculation,
                       reduceBlockDim,
                       reduceThreadDim,
                       threadsPerBlock * sizeof(double),
                       0,
                       nKcells,
                       devBotRows,
                       devSource,
                       devDiag,
                       devRightCols,
                       devCornerVals,
                       devLastSources,
                       devXl);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(
      xlValues.data(), devXl, nCells * sizeof(double), hipMemcpyDeviceToHost));

    hipLaunchKernelGGL(xiCalculation,
                       blockDim,
                       threadDim,
                       0,
                       0,
                       nKcells,
                       numRows,
                       devXl,
                       devRightCols,
                       devDiag,
                       devSource,
                       devXi);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(
      hipMemcpy(xiValues.data(), devXi, numBytes, hipMemcpyDeviceToHost));
  }

#ifdef DIAGNOSTIC
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nKcells; ++k) {
      int const i = c * nKcells + k;
      std::cout << "[" << c << ", " << k << "]: Expected " << xiExpected[i]
                << ", Actual: " << xiValues[i] << std::endl;
    }

    std::cout << "[" << c << ", " << nKcells << "]: Expected " << xlExpected[c]
              << ", Actual: " << xlValues[c] << std::endl;
  }
#endif
}