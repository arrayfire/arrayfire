/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <transpose.hpp>

#include <utility>
#include <cassert>

using af::dim4;

namespace cpu
{

static inline unsigned getIdx(const dim4 &strides,
        int i, int j = 0, int k = 0, int l = 0)
{
    return (l * strides[3] +
            k * strides[2] +
            j * strides[1] +
            i );
}

template<typename T>
T getConjugate(const T &in)
{
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in)
{
    return std::conj(in);
}

template<>
cdouble getConjugate(const cdouble &in)
{
    return std::conj(in);
}

template<typename T, bool conjugate>
void transpose_(T *out, const T *in, const af::dim4 &odims, const af::dim4 &idims,
                const af::dim4 &ostrides, const af::dim4 &istrides)
{
    for (dim_t l = 0; l < odims[3]; ++l) {
        for (dim_t k = 0; k < odims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once
            for (dim_t j = 0; j < odims[1]; ++j) {
                for (dim_t i = 0; i < odims[0]; ++i) {
                    // calculate array indices based on offsets and strides
                    // the helper getIdx takes care of indices
                    const dim_t inIdx  = getIdx(istrides,j,i,k,l);
                    const dim_t outIdx = getIdx(ostrides,i,j,k,l);
                    if(conjugate)
                        out[outIdx] = getConjugate(in[inIdx]);
                    else
                        out[outIdx] = in[inIdx];
                }
            }
            // outData and inData pointers doesn't need to be
            // offset as the getIdx function is taking care
            // of the batch parameter
        }
    }
}

template<typename T>
Array<T> transpose(const Array<T> &in, const bool conjugate)
{
    const dim4 inDims = in.dims();

    dim4 outDims   = dim4(inDims[1],inDims[0],inDims[2],inDims[3]);

    // create an array with first two dimensions swapped
    Array<T> out  = createEmptyArray<T>(outDims);

    // get data pointers for input and output Arrays
    T* outData          = out.get();
    const T*   inData   = in.get();

    if(conjugate) {
        transpose_<T, true>(outData, inData,
                            out.dims(), in.dims(), out.strides(), in.strides());
    } else {
        transpose_<T, false>(outData, inData,
                             out.dims(), in.dims(), out.strides(), in.strides());
    }

    return out;
}

template<typename T, bool conjugate>
void transpose_inplace(T *in, const af::dim4 &idims, const af::dim4 &istrides)
{
    for (dim_t l = 0; l < idims[3]; ++l) {
        for (dim_t k = 0; k < idims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once
            //
            // Run only bottom triangle. std::swap swaps with upper triangle
            for (dim_t j = 0; j < idims[1]; ++j) {
                for (dim_t i = j + 1; i < idims[0]; ++i) {
                    // calculate array indices based on offsets and strides
                    // the helper getIdx takes care of indices
                    const dim_t iIdx  = getIdx(istrides,j,i,k,l);
                    const dim_t oIdx = getIdx(istrides,i,j,k,l);
                    if(conjugate) {
                        in[iIdx] = getConjugate(in[iIdx]);
                        in[oIdx] = getConjugate(in[oIdx]);
                        std::swap(in[iIdx], in[oIdx]);
                    }
                    else {
                        std::swap(in[iIdx], in[oIdx]);
                    }
                }
            }
        }
    }
}

template<typename T>
void transpose_inplace(Array<T> &in, const bool conjugate)
{
    // get data pointers for input and output Arrays
    T* inData = in.get();

    if(conjugate) {
        transpose_inplace<T, true >(inData, in.dims(), in.strides());
    } else {
        transpose_inplace<T, false>(inData, in.dims(), in.strides());
    }
}

#define INSTANTIATE(T)                                                      \
    template Array<T> transpose(const Array<T> &in, const bool conjugate);  \
    template void transpose_inplace(Array<T> &in, const bool conjugate);

INSTANTIATE(float  )
INSTANTIATE(cfloat )
INSTANTIATE(double )
INSTANTIATE(cdouble)
INSTANTIATE(char   )
INSTANTIATE(int    )
INSTANTIATE(uint   )
INSTANTIATE(uchar  )
INSTANTIATE(intl   )
INSTANTIATE(uintl  )


}
