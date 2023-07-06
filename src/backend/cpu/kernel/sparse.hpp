/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <kernel/sort_helper.hpp>
#include <math.hpp>
#include <utility.hpp>
#include <algorithm>
#include <tuple>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void coo2dense(Param<T> output, CParam<T> values, CParam<int> rowIdx,
               CParam<int> colIdx) {
    const T *vPtr   = values.get();
    const int *rPtr = rowIdx.get();
    const int *cPtr = colIdx.get();

    T *outPtr = output.get();

    af::dim4 ostrides = output.strides();

    int nNZ = values.dims(0);
    for (int i = 0; i < nNZ; i++) {
        T v   = vPtr[i];
        int r = rPtr[i];
        int c = cPtr[i];

        int offset = r + c * ostrides[1];

        outPtr[offset] = v;
    }
}

  template <typename T>
  void printArray(const char *name, const unsigned N, const T *thing) {
      printf("%s:\n", name);
      for (int i = 0; i < N; i++)
        printf("%f, ", thing[i]);
      printf("\n\n");
  }

  template <>
  void printArray(const char *name, const unsigned N, const int *thing) {
      printf("%s:\n", name);
      for (int i = 0; i < N; i++)
        printf("%d, ", thing[i]);
      printf("\n\n");
  }

template<typename T>
void dense2csr(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
               CParam<T> in) {
    const T *iPtr = in.get();
    T *vPtr       = values.get();
    int *rPtr     = rowIdx.get();
    int *cPtr     = colIdx.get();

    int stride    = in.strides(1);
    af::dim4 dims = in.dims();

    int offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
        rPtr[i] = offset;
        for (int j = 0; j < dims[1]; ++j) {
            if (iPtr[j * stride + i] != scalar<T>(0)) {
                vPtr[offset]   = iPtr[j * stride + i];
                cPtr[offset++] = j;
            }
        }
    }
    rPtr[dims[0]] = offset;

    printArray("rPtr", values.dims()[0], rPtr);
    printArray("cPtr", values.dims()[0], cPtr);
    printArray("vPtr", values.dims()[0], vPtr);
}

template<typename T>
void csr2dense(Param<T> out, CParam<T> values, CParam<int> rowIdx,
               CParam<int> colIdx) {
    T *oPtr         = out.get();
    const T *vPtr   = values.get();
    const int *rPtr = rowIdx.get();
    const int *cPtr = colIdx.get();

    int stride = out.strides(1);

    int r = rowIdx.dims(0);
    for (int i = 0; i < r - 1; i++) {
        for (int ii = rPtr[i]; ii < rPtr[i + 1]; ++ii) {
            int j                = cPtr[ii];
            T v                  = vPtr[ii];
            oPtr[j * stride + i] = v;
        }
    }
}

// Modified code from sort helper
template<typename T>
using SpKeyIndexPair =
    std::tuple<int, T, int>;  // sorting index, value, other index

template<typename T>
struct SpKIPCompareK {
    bool operator()(const SpKeyIndexPair<T> &lhs,
                    const SpKeyIndexPair<T> &rhs) {
        int lhsVal = std::get<0>(lhs);
        int rhsVal = std::get<0>(rhs);
        // Always returns ascending
        return (lhsVal < rhsVal);
    }
};

template<typename T>
void csr2coo(Param<T> ovalues, Param<int> orowIdx, Param<int> ocolIdx,
             CParam<T> ivalues, CParam<int> irowIdx, CParam<int> icolIdx) {
    // First calculate the linear index
    T *ovPtr   = ovalues.get();
    int *orPtr = orowIdx.get();
    int *ocPtr = ocolIdx.get();

    const T *ivPtr   = ivalues.get();
    const int *irPtr = irowIdx.get();
    const int *icPtr = icolIdx.get();

    // Create cordinate form of the row array
    for (int i = 0; i < (int)irowIdx.dims().elements() - 1; i++) {
        std::fill_n(orPtr + irPtr[i], irPtr[i + 1] - irPtr[i], i);
    }

    // Sort the coordinate form using column index
    // Uses code from sort_by_key kernels
    typedef SpKeyIndexPair<T> CurrentPair;
    int size = ovalues.dims(0);
    std::vector<CurrentPair> pairKeyVal(size);

    for (int x = 0; x < size; x++) {
        pairKeyVal[x] = std::make_tuple(icPtr[x], ivPtr[x], orPtr[x]);
    }

    std::stable_sort(pairKeyVal.begin(), pairKeyVal.end(), SpKIPCompareK<T>());

    for (int x = 0; x < (int)ovalues.dims().elements(); x++) {
        std::tie(ocPtr[x], ovPtr[x], orPtr[x]) = pairKeyVal[x];
    }
}

template<typename T>
void coo2csr(Param<T> ovalues, Param<int> orowIdx, Param<int> ocolIdx,
             CParam<T> ivalues, CParam<int> irowIdx, CParam<int> icolIdx) {
    T *ovPtr   = ovalues.get();
    int *orPtr = orowIdx.get();
    int *ocPtr = ocolIdx.get();

    const T *ivPtr   = ivalues.get();
    const int *irPtr = irowIdx.get();
    const int *icPtr = icolIdx.get();

    // Sort the colidx and values based on rowIdx
    // Uses code from sort_by_key kernels
    typedef SpKeyIndexPair<T> CurrentPair;
    int size = ovalues.dims(0);
    std::vector<CurrentPair> pairKeyVal(size);

    for (int x = 0; x < size; x++) {
        pairKeyVal[x] = std::make_tuple(irPtr[x], ivPtr[x], icPtr[x]);
    }

    std::stable_sort(pairKeyVal.begin(), pairKeyVal.end(), SpKIPCompareK<T>());

    ovPtr[0] = 0;
    for (int x = 0; x < (int)ovalues.dims().elements(); x++) {
        int row = -2;  // Some value that will make orPtr[row + 1] error out
        std::tie(row, ovPtr[x], ocPtr[x]) = pairKeyVal[x];
        orPtr[row + 1]++;
    }

    // Compress row storage
    for (int x = 1; x < (int)orowIdx.dims().elements(); x++) {
        orPtr[x] += orPtr[x - 1];
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
