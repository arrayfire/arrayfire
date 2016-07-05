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
struct dns_csr
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
struct csr_dns
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
        for (int i = 0; i < r; i++) {
            for (int ii = rPtr[i]; ii < rPtr[i+1]; ++ii) {
                int j = cPtr[ii];
                T v = vPtr[ii];
                oPtr[j*stride + i] = v;
            }
        }
    }
};

}
}
