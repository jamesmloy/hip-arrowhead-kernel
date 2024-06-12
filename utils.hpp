#pragma once

#include "hip/hip_runtime.h"
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    hipError_t status = command;                                               \
    if (status != hipSuccess) {                                                \
      std::cerr << "Error: HIP reports " << hipGetErrorString(status)          \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
  }

hipDeviceProp_t
getDeviceProps(int const deviceId)
{
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  return props;
}

std::ostream&
operator<<(std::ostream& o, hipDeviceProp_t const& props)
{
  o << "Device properties:\n"
    << "Total Global Mem (bytes): " << props.totalGlobalMem << "\n"
    << "Shared memory per block: " << props.sharedMemPerBlock << "\n"
    << "Warp Size: " << props.warpSize << "\n"
    << "Max threads per block: " << props.maxThreadsPerBlock << "\n"
    << "Max threads in each dimension: [" << props.maxThreadsDim[0] << ", "
    << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << "]\n"
    << "Max grid size: [" << props.maxGridSize[0] << ", "
    << props.maxGridSize[1] << ", " << props.maxGridSize[2] << "]\n";
  return o;
}

template<typename T>
class ContiguousVecOfVec
{
public:
  ContiguousVecOfVec(int nr, int nc)
    : _nr(nr)
    , _nc(nc)
    , _totSize(nr * nc)
    , _data(_totSize)
  {
  }

  template<typename RndGen>
  void randomize(RndGen& gen, T const& minVal, T const& maxVal)
  {
    std::uniform_real_distribution<T> dis(minVal, maxVal);
    for (int i = 0; i < _totSize; ++i)
      _data[i] = dis(gen);
  }

  void setAllTo(T const val)
  {
    for (int i = 0; i < _totSize; ++i) _data[i] = val;
  }

  T* data() { return _data.data(); }

  std::pair<T*, T*> row(int i)
  {
    auto beg = _data.data() + i * _nc;
    auto end = beg + _nc;
    return std::make_pair(beg, end);
  }

  T const& operator[](int const i) const { return _data[i]; }
  T& operator[](int const i) { return _data[i]; }

  size_t numBytes() const { return _totSize * sizeof(T); }

private:
  int _nr;
  int _nc;
  int _totSize;
  std::vector<T> _data;
};

template<typename RngType>
class ArrowheadView
{
public:
  using DType = std::remove_reference_t<decltype(*(std::declval<RngType>().first))>;

  ArrowheadView(RngType diag, RngType rightCol, RngType botRow, DType corner)
    : _diag(std::move(diag))
    , _rightCol(std::move(rightCol))
    , _botRow(std::move(botRow))
    , _corner(corner)
    , _rows(std::distance(diag.first, diag.second) + 1)
  {
  }

  std::vector<DType> multiply(RngType x) const
  {
    std::vector<DType> result(_rows, 0);
    int const lastRow = _rows - 1;
    for (int i = 0; i < _rows - 1; ++i)
    {
      result[i] = diag(i) * (*(x.first + i)) + rightCol(i) * (*(x.second - 1));
      result[lastRow] += botRow(i) * (*(x.first + i));
    }
    result[lastRow] += _corner * (*(x.second - 1));

    return result;
  }

  DType const& diag(int const i) const { return *(_diag.first + i); }
  DType const& rightCol(int const i) const { return *(_rightCol.first + i); }
  DType const& botRow(int const i) const { return *(_botRow.first + i); }

private:
  RngType _diag;
  RngType _rightCol;
  RngType _botRow;
  DType _corner;
  int _rows;
};


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


#if 1
template <typename T>
class ArrowheadSystem
{
public:
  using VecOfVec = ContiguousVecOfVec<T>;

  template <typename RndGen>
  ArrowheadSystem(int const nCells, int const nKcells, RndGen &gen, T const minVal = 0.5, T const maxVal = 1.5)
    : _nCells(nCells)
    , _nKcells(nKcells)
    , _diags(nCells, nKcells)
    , _rightCols(nCells, nKcells)
    , _botRows(nCells, nKcells)
    , _cornerVals()
    , _sources(nCells, nKcells)
    , _lastSource(nCells, 0)
    , _xlExpected(nCells, 0)
    , _xiExpected(nCells, nKcells)
  {
    _diags.randomize(gen, minVal, maxVal);

    // right columns, set these as opposite the
    // diagonal
    for (int i = 0; i < nCells * nKcells; ++i)
      _rightCols[i] = -_diags[i] * 0.5;

    _botRows.randomize(gen, minVal, maxVal);

    _cornerVals.reserve(nCells);
    for (int i = 0; i < nCells; ++i) {
      auto begEnd = _botRows.row(i);
      _cornerVals.push_back(-std::accumulate(begEnd.first, begEnd.second, 0.0));
    }

    _xiExpected.randomize(gen, minVal, maxVal);
    std::uniform_real_distribution<double> dis(0.5, 1.5);
    for (int i = 0; i < nCells; ++i)
      _xlExpected[i] = dis(gen);

    for (int c = 0; c < nCells; ++c) {
      for (int k = 0; k < nKcells; ++k) {
        int const i = c * nKcells + k;
        _sources[i] = _diags[i] * _xiExpected[i] + _rightCols[i] * _xlExpected[c];
        _lastSource[c] += _botRows[i] * _xiExpected[i];
      }
      _lastSource[c] += _cornerVals[c] * _xlExpected[c];
    }

    allocateDeviceMemory();
  }

  ~ArrowheadSystem()
  {
    // diagonals
    hipFree(_devDiag);
    // right columns
    hipFree(_devRightCols);
    // bottom rows
    hipFree(_devBotRows);
    // source term
    hipFree(_devSource);
    // result vector for the second result
    hipFree(_devXi);
    // corner value of the matrix
    hipFree(_devCornerVals);
    // the last source term
    hipFree(_devLastSources);
    // result vector for the first result
    hipFree(_devXl);
  }

  void solve(std::vector<double> &xl, VecOfVec &xi)
  {
    int const threadsPerBlock = 128;
    int const numRows = _nCells * _nKcells;
    int const blocks = (numRows + threadsPerBlock - 1) / threadsPerBlock;
    dim3 reduceBlockDim(_nCells, 1, 1);
    dim3 reduceThreadDim(threadsPerBlock, 1, 1);
    dim3 blockDim(blocks, 1, 1);
    dim3 threadDim(threadsPerBlock, 1, 1 );

    auto const numBytes = _diags.numBytes();

    hipLaunchKernelGGL(xlCalculation,
                       reduceBlockDim,
                       reduceThreadDim,
                       threadsPerBlock * sizeof(T),
                       0,
                       _nKcells,
                       _devBotRows,
                       _devSource,
                       _devDiag,
                       _devRightCols,
                       _devCornerVals,
                       _devLastSources,
                       _devXl);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(
      xl.data(), _devXl, _nCells * sizeof(T), hipMemcpyDeviceToHost));

    hipLaunchKernelGGL(xiCalculation,
                       blockDim,
                       threadDim,
                       0,
                       0,
                       _nKcells,
                       numRows,
                       _devXl,
                       _devRightCols,
                       _devDiag,
                       _devSource,
                       _devXi);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(
      hipMemcpy(xi.data(), _devXi, numBytes, hipMemcpyDeviceToHost));
  }

  std::vector<T> const& xlExpected() const
  { return _xlExpected; }
  VecOfVec const& xiExpected() const
  { return _xiExpected; }

private:

  void allocateDeviceMemory()
  {
    auto const numBytes = _diags.numBytes();

    // diagonals
    HIP_CHECK(hipMalloc(&_devDiag, numBytes));
    HIP_CHECK(
      hipMemcpy(_devDiag, _diags.data(), numBytes, hipMemcpyHostToDevice));

    // right columns
    HIP_CHECK(hipMalloc(&_devRightCols, numBytes));
    HIP_CHECK(hipMemcpy(
      _devRightCols, _rightCols.data(), numBytes, hipMemcpyHostToDevice));

    // bottom rows
    HIP_CHECK(hipMalloc(&_devBotRows, numBytes));
    HIP_CHECK(
      hipMemcpy(_devBotRows, _botRows.data(), numBytes, hipMemcpyHostToDevice));

    // source term
    HIP_CHECK(hipMalloc(&_devSource, numBytes));
    HIP_CHECK(
      hipMemcpy(_devSource, _sources.data(), numBytes, hipMemcpyHostToDevice));

    // corner value of the matrix
    HIP_CHECK(hipMalloc(&_devCornerVals, _nCells * sizeof(T)));
    HIP_CHECK(hipMemcpy(_devCornerVals,
                        _cornerVals.data(),
                        _nCells * sizeof(T),
                        hipMemcpyHostToDevice));

    // the last source term
    HIP_CHECK(hipMalloc(&_devLastSources, _nCells * sizeof(T)));
    HIP_CHECK(hipMemcpy(_devLastSources,
                        _lastSource.data(),
                        _nCells * sizeof(T),
                        hipMemcpyHostToDevice));

    // result vector for the second result
    HIP_CHECK(hipMalloc(&_devXi, numBytes));
    // HIP_CHECK(hipMemcpy(_devXi, xi.data(), numBytes, hipMemcpyHostToDevice));

    // result vector for the first result
    HIP_CHECK(hipMalloc(&_devXl, _nCells * sizeof(T)));
    // HIP_CHECK(hipMemcpy(_devXl,
    //                     xl.data(),
    //                     _nCells * sizeof(T),
    //                     hipMemcpyHostToDevice));
  }

  int const _nCells;
  int const _nKcells;

  //
  // HOST MEMORY
  //
  // matrix coefficients
  VecOfVec _diags;
  VecOfVec _rightCols;
  VecOfVec _botRows;
  std::vector<T> _cornerVals;

  // source terms
  VecOfVec _sources;
  std::vector<T> _lastSource;

  // expected solution
  std::vector<T> _xlExpected;
  VecOfVec _xiExpected;

  //
  // DEVICE MEMORY
  //
  // diagonals
  T* _devDiag = nullptr;
  // right columns
  T* _devRightCols = nullptr;
  // bottom rows
  T* _devBotRows = nullptr;
  // source term
  T* _devSource = nullptr;
  // result vector for the second result
  T* _devXi = nullptr;
  // corner value of the matrix
  T* _devCornerVals = nullptr;
  // the last source term
  T* _devLastSources = nullptr;
  // result vector for the first result
  T* _devXl = nullptr;
};
#endif