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
#include <utility.hpp>
#include <err_cpu.hpp>

namespace cpu
{
namespace kernel
{

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
void transpose(Param<T> output, CParam<T> input)
{
    const dim4 odims    = output.dims();
    const dim4 ostrides = output.strides();
    const dim4 istrides = input.strides();

    T * out = output.get();
    T const * const in = input.get();

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
void transpose(Param<T> out, CParam<T> in, const bool conjugate)
{
    return (conjugate ? transpose<T, true>(out, in) : transpose<T, false>(out, in));
}

template<typename T, bool conjugate>
void transpose_inplace(Param<T> input)
{
    const dim4 idims    = input.dims();
    const dim4 istrides = input.strides();

    T * in = input.get();

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
void transpose_inplace(Param<T> in, const bool conjugate)
{
    return (conjugate ? transpose_inplace<T, true >(in) : transpose_inplace<T, false>(in));
}

}
}
