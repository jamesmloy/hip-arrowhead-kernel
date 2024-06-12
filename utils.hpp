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