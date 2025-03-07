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
#include <err_cpu.hpp>
#include <utility.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
T getConjugate(const T &in) {
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in) {
    return std::conj(in);
}

template<>
cdouble getConjugate(const cdouble &in) {
    return std::conj(in);
}

template<typename T, int M, int N>
void transpose_kernel(T *output, const T *input, int ostride, int istride) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) { output[i * ostride] = input[i]; }
        input += istride;
        output++;
    }
}

template<typename T>
void transpose_real(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *out            = output.get();
    T const *const in = input.get();

    constexpr int M = 8;
    constexpr int N = 8;

    dim_t odims1_down = floor(odims[1] / N) * N;
    dim_t odims0_down = floor(odims[0] / M) * M;

    for (dim_t l = 0; l < odims[3]; ++l) {
        for (dim_t k = 0; k < odims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once
            T *out_      = out + l * ostrides[3] + k * ostrides[2];
            const T *in_ = in + l * istrides[3] + k * istrides[2];

            if (odims1_down > 0) {
                for (dim_t j = 0; j <= odims1_down; j += N) {
                    for (dim_t i = 0; i < odims0_down; i += M) {
                        transpose_kernel<T, M, N>(out_, in_, ostrides[1],
                                                  istrides[1]);
                        out_ += M;
                        in_ += istrides[1] * N;
                    }

                    for (dim_t jj = 0; jj < N; jj++) {
                        for (dim_t i = odims0_down; i < odims[0]; i++) {
                            *out_ = *in_;
                            out_++;
                            in_ += istrides[1];
                        }
                        out_ += ostrides[1] - (odims[0] - odims0_down);
                        in_ -= (odims[0] - odims0_down) * istrides[1] - 1;
                    }
                    out_ = out + l * ostrides[3] + k * ostrides[2] +
                           j * ostrides[1];
                    in_ = in + l * istrides[3] + k * istrides[2] + j;
                }
            }
            for (dim_t j = odims1_down; j < odims[1]; j++) {
                out_ =
                    out + l * ostrides[3] + k * ostrides[2] + j * ostrides[1];
                in_ = in + l * istrides[3] + k * istrides[2] + j;
                for (dim_t i = 0; i < odims[0]; i++) {
                    *out_ = *in_;
                    out_++;
                    in_ += istrides[1];
                }
            }
        }
    }
}

template<typename T>
void transpose_conj(Param<T> output, CParam<T> input) {
    const af::dim4 odims    = output.dims();
    const af::dim4 ostrides = output.strides();
    const af::dim4 istrides = input.strides();

    T *out            = output.get();
    T const *const in = input.get();

    for (dim_t l = 0; l < odims[3]; ++l) {
        for (dim_t k = 0; k < odims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once

            for (dim_t j = 0; j < odims[1]; ++j) {
                for (dim_t i = 0; i < odims[0]; ++i) {
                    // calculate array indices based on offsets and strides
                    // the helper getIdx takes care of indices
                    const dim_t inIdx  = getIdx(istrides, j, i, k, l);
                    const dim_t outIdx = getIdx(ostrides, i, j, k, l);
                    out[outIdx]        = getConjugate(in[inIdx]);
                }
            }
            // outData and inData pointers doesn't need to be
            // offset as the getIdx function is taking care
            // of the batch parameter
        }
    }
}

template<typename T>
void transpose(Param<T> out, CParam<T> in, const bool conjugate) {
    return (conjugate ? transpose_conj<T>(out, in)
                      : transpose_real<T>(out, in));
}

template<typename T, bool conjugate>
void transpose_inplace(Param<T> input) {
    const af::dim4 idims    = input.dims();
    const af::dim4 istrides = input.strides();

    T *in = input.get();

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
                    const dim_t iIdx = getIdx(istrides, j, i, k, l);
                    const dim_t oIdx = getIdx(istrides, i, j, k, l);
                    if (conjugate) {
                        in[iIdx] = getConjugate(in[iIdx]);
                        in[oIdx] = getConjugate(in[oIdx]);
                        std::swap(in[iIdx], in[oIdx]);
                    } else {
                        std::swap(in[iIdx], in[oIdx]);
                    }
                }
            }
        }
    }
}

template<typename T>
void transpose_inplace(Param<T> in, const bool conjugate) {
    return (conjugate ? transpose_inplace<T, true>(in)
                      : transpose_inplace<T, false>(in));
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
