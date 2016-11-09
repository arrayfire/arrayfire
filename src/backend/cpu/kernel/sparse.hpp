/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <utility.hpp>
#include <math.hpp>
#include <kernel/sort_helper.hpp>
#include <algorithm>

namespace cpu
{
namespace kernel
{

template<typename T>
void coo2dense(Array<T> output,
               Array<T> const values, Array<int> const rowIdx, Array<int> const colIdx)
{
    T   const * const vPtr = values.get();
    int const * const rPtr = rowIdx.get();
    int const * const cPtr = colIdx.get();

    T * outPtr = output.get();

    af::dim4 ostrides = output.strides();

    int nNZ = values.dims()[0];
    for(int i = 0; i < nNZ; i++) {
        T   v = vPtr[i];
        int r = rPtr[i];
        int c = cPtr[i];

        int offset = r + c * ostrides[1];

        outPtr[offset] = v;
    }
}

template<typename T>
struct dense_csr
{
    void operator()(Array<T> values, Array<int> rowIdx, Array<int> colIdx,
                    Array<T> const in)
    {
        T const * const iPtr = in.get();
        T       * const vPtr = values.get();
        int     * const rPtr = rowIdx.get();
        int     * const cPtr = colIdx.get();

        int stride = in.strides()[1];
        af::dim4 dims = in.dims();

        int offset = 0;
        for (int i = 0; i < dims[0]; ++i) {
            rPtr[i] = offset;
            for (int j = 0; j < dims[1]; ++j) {
                if (iPtr[j*stride + i] != scalar<T>(0)) {
                    vPtr[offset] = iPtr[j*stride + i];
                    cPtr[offset++] = j;
                }
            }
        }
        rPtr[dims[0]] = offset;
    }
};

template<typename T>
struct csr_dense
{
    void operator()(Array<T> out,
                    Array<T> const values, Array<int> const rowIdx, Array<int> const colIdx)
    {
        T         * const oPtr = out.get();
        T   const * const vPtr = values.get();
        int const * const rPtr = rowIdx.get();
        int const * const cPtr = colIdx.get();

        int stride = out.strides()[1];

        int r = rowIdx.dims()[0];
        for (int i = 0; i < r - 1; i++) {
            for (int ii = rPtr[i]; ii < rPtr[i+1]; ++ii) {
                int j = cPtr[ii];
                T v = vPtr[ii];
                oPtr[j*stride + i] = v;
            }
        }
    }
};

// Modified code from sort helper
template <typename T>
using SpKeyIndexPair = std::tuple<int, T, int>; // sorting index, value, other index

template <typename T>
struct SpKIPCompareK
{
    bool operator()(const SpKeyIndexPair<T> &lhs, const SpKeyIndexPair<T> &rhs)
    {
        int lhsVal = std::get<0>(lhs);
        int rhsVal = std::get<0>(rhs);
        // Always returns ascending
        return (lhsVal < rhsVal);
    }
};

template<typename T>
struct csr_coo
{
    void operator()(Array<T> ovalues, Array<int> orowIdx, Array<int> ocolIdx,
                    Array<T> const ivalues, Array<int> const irowIdx, Array<int> const icolIdx)
    {
        // First calculate the linear index
        T         * const ovPtr = ovalues.get();
        int       * const orPtr = orowIdx.get();
        int       * const ocPtr = ocolIdx.get();

        T   const * const ivPtr = ivalues.get();
        int const * const irPtr = irowIdx.get();
        int const * const icPtr = icolIdx.get();

        // Create cordinate form of the row array
        for(int i = 0; i < (int)irowIdx.elements() - 1; i++) {
            std::fill_n(orPtr + irPtr[i], irPtr[i + 1] - irPtr[i], i);
        }

        // Sort the coordinate form using column index
        // Uses code from sort_by_key kernels
        typedef SpKeyIndexPair<T> CurrentPair;
        int size = ovalues.dims()[0];
        size_t bytes = size * sizeof(CurrentPair);
        CurrentPair *pairKeyVal = (CurrentPair *)memAlloc<char>(bytes);

        for(int x = 0; x < size; x++) {
           pairKeyVal[x] = std::make_tuple(icPtr[x], ivPtr[x], orPtr[x]);
        }

        std::stable_sort(pairKeyVal, pairKeyVal + size, SpKIPCompareK<T>());

        for(int x = 0; x < (int)ovalues.elements(); x++) {
            ocPtr[x] = std::get<0>(pairKeyVal[x]);
            ovPtr[x] = std::get<1>(pairKeyVal[x]);
            orPtr[x] = std::get<2>(pairKeyVal[x]);
        }

        memFree((char *)pairKeyVal);
    }
};

template<typename T>
struct coo_csr
{
    void operator()(Array<T> ovalues, Array<int> orowIdx, Array<int> ocolIdx,
                    Array<T> const ivalues, Array<int> const irowIdx, Array<int> const icolIdx)
    {
        T         * const ovPtr = ovalues.get();
        int       * const orPtr = orowIdx.get();
        int       * const ocPtr = ocolIdx.get();

        T   const * const ivPtr = ivalues.get();
        int const * const irPtr = irowIdx.get();
        int const * const icPtr = icolIdx.get();

        // Sort the colidx and values based on rowIdx
        // Uses code from sort_by_key kernels
        typedef SpKeyIndexPair<T> CurrentPair;
        int size = ovalues.dims()[0];
        size_t bytes = size * sizeof(CurrentPair);
        CurrentPair *pairKeyVal = (CurrentPair *)memAlloc<char>(bytes);

        for(int x = 0; x < size; x++) {
           pairKeyVal[x] = std::make_tuple(irPtr[x], ivPtr[x], icPtr[x]);
        }

        std::stable_sort(pairKeyVal, pairKeyVal + size, SpKIPCompareK<T>());

        ovPtr[0] = 0;
        for(int x = 0; x < (int)ovalues.elements(); x++) {
            int row  = std::get<0>(pairKeyVal[x]);
            ovPtr[x] = std::get<1>(pairKeyVal[x]);
            ocPtr[x] = std::get<2>(pairKeyVal[x]);
            orPtr[row + 1]++;
        }

        // Compress row storage
        for(int x = 1; x < (int)orowIdx.elements(); x++) {
            orPtr[x] += orPtr[x - 1];
        }

        memFree((char *)pairKeyVal);
    }
};

}
}
