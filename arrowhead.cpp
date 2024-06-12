#include "hip/hip_runtime.h"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "utils.hpp"

std::random_device rd;
std::mt19937 gen(rd());

#define HIP_ENABLE_PRINTF

__global__ void
xlCalculation(int const valsPerBlock,
              double const* botRows,    // ncells * nkcells
              double const* sources,    // ncells * nkcells
              double const* diags,      // ncells * nkcells
              double const* rightCols,  // ncells * nkcells
              double const* cornerVals, // ncells
              double const* lastSource, // ncells
              double* xl)               // ncells, output
{
  HIP_DYNAMIC_SHARED(double, toSum);

  int const tid = threadIdx.x;
  int const iterations = (valsPerBlock + blockDim.x - 1) / blockDim.x;
  int const globIdxStart = valsPerBlock * blockIdx.x;

  double num = lastSource[blockIdx.x];
  double den = cornerVals[blockIdx.x];

  // sum numerator
  for (int it = 0; it < iterations; ++it) {
    int const countOnBlock = tid + blockDim.x * it;
    int const globIdx = globIdxStart + countOnBlock;
    int const inRange = int(countOnBlock < valsPerBlock);

    // load values to local memory, masking remainder
    toSum[tid] = botRows[globIdx] * sources[globIdx] / diags[globIdx] * inRange;
    __syncthreads();
    if (inRange) {
      // tree based sum
      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          toSum[tid] += toSum[tid + s];
        }
        __syncthreads();
      }
    }
    if (tid == 0) {
      num -= toSum[0];
      // printf("num %f\n", num);
    }
  }


  // sum denomenator
  for (int it = 0; it < iterations; ++it) {
    int const countOnBlock = tid + blockDim.x * it;
    int const globIdx = globIdxStart + countOnBlock;
    int const inRange = int(countOnBlock < valsPerBlock);

    // load values to local memory, masking remainder
    toSum[tid] =
      botRows[globIdx] * rightCols[globIdx] / diags[globIdx] * inRange;
    __syncthreads();
    if (inRange) {
      // tree based sum
      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          toSum[tid] += toSum[tid + s];
        }
        __syncthreads();
      }
    }
    if (tid == 0) {
      den -= toSum[0];
      // printf("den %f\n", den);
    }
  }


  if (tid == 0)
    xl[blockIdx.x] = num / den;
}

__global__ void
xiCalculation(int const valsPerCell,
              int const totVals,
              double const* xl,        // ncells
              double const* rightCols, // ncells * nkcells
              double const* diags,     // ncells * nkcells
              double const* sources,   // ncells * nkcells
              double* xi)              // ncells * nkcells, output
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  int const cellIdx = i / valsPerCell;

  if (i < totVals) {
    xi[i] = sources[i] / diags[i] - rightCols[i] * xl[cellIdx] / diags[i];
  }
}

using VecOfVec = ContiguousVecOfVec<double>;

int
main()
{
  int const nCells = 2;
  int const nKcells = 100;
  int const deviceId = 0;
  int const threadsPerBlock = 128;
  int const numRows = nCells * nKcells;
  int const blocks = (numRows + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blockDim{ blocks, 1, 1 };
  dim3 threadDim{ threadsPerBlock, 1, 1 };
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

  // for (int c = 0; c < nCells; ++c) {
  //   for (int k = 0; k < nKcells; ++k) {
  //     int const i = c * nKcells + k;
  //     std::cout << "diag: " << diags[i] << ", right: " << rightCols[i]
  //               << ", bot: " << botRows[i] << ", source: " << source[i]
  //               << ", xi: " << xiExpected[i] << "\n";
  //   }
  //   std::cout << "corner: " << cornerVals[c]
  //             << ", last source: " << lastSource[c] << ", xl: " << xlExpected[c]
  //             << "\n";
  // }

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
    dim3 reduceBlockDim{ nCells, 1, 1 };
    dim3 reduceThreadDim{ threadsPerBlock, 1, 1 };
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

  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nKcells; ++k) {
      int const i = c * nKcells + k;
      std::cout << "[" << c << ", " << k << "]: Expected " << xiExpected[i]
                << ", Actual: " << xiValues[i] << std::endl;
    }

    std::cout << "[" << c << ", " << nKcells << "]: Expected " << xlExpected[c]
              << ", Actual: " << xlValues[c] << std::endl;
  }
}