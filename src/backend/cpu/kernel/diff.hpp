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

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void diff1(Param<T> out, CParam<T> in, int const dim) {
    af::dim4 dims = out.dims();
    // Bool for dimension
    bool is_dim0 = dim == 0;
    bool is_dim1 = dim == 1;
    bool is_dim2 = dim == 2;
    bool is_dim3 = dim == 3;

    T const* const inPtr = in.get();
    T* outPtr            = out.get();

    // TODO: Improve this
    for (dim_t l = 0; l < dims[3]; l++) {
        for (dim_t k = 0; k < dims[2]; k++) {
            for (dim_t j = 0; j < dims[1]; j++) {
                for (dim_t i = 0; i < dims[0]; i++) {
                    // Operation: out[index] = in[index + 1 * dim_size] -
                    // in[index]
                    int idx     = getIdx(in.strides(), i, j, k, l);
                    int jdx     = getIdx(in.strides(), i + is_dim0, j + is_dim1,
                                         k + is_dim2, l + is_dim3);
                    int odx     = getIdx(out.strides(), i, j, k, l);
                    outPtr[odx] = inPtr[jdx] - inPtr[idx];
                }
            }
        }
    }
}

template<typename T>
void diff2(Param<T> out, CParam<T> in, int const dim) {
    af::dim4 dims = out.dims();
    // Bool for dimension
    bool is_dim0 = dim == 0;
    bool is_dim1 = dim == 1;
    bool is_dim2 = dim == 2;
    bool is_dim3 = dim == 3;

    T const* const inPtr = in.get();
    T* outPtr            = out.get();

    // TODO: Improve this
    for (dim_t l = 0; l < dims[3]; l++) {
        for (dim_t k = 0; k < dims[2]; k++) {
            for (dim_t j = 0; j < dims[1]; j++) {
                for (dim_t i = 0; i < dims[0]; i++) {
                    // Operation: out[index] = in[index + 1 * dim_size] -
                    // in[index]
                    int idx = getIdx(in.strides(), i, j, k, l);
                    int jdx = getIdx(in.strides(), i + is_dim0, j + is_dim1,
                                     k + is_dim2, l + is_dim3);
                    int kdx =
                        getIdx(in.strides(), i + 2 * is_dim0, j + 2 * is_dim1,
                               k + 2 * is_dim2, l + 2 * is_dim3);
                    int odx = getIdx(out.strides(), i, j, k, l);
                    outPtr[odx] =
                        inPtr[kdx] + inPtr[idx] - inPtr[jdx] - inPtr[jdx];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
