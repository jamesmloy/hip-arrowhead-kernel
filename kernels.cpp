#include "hip/hip_runtime.h"

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