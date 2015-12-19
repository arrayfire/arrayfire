/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{

using af::dim4;

template<typename T>
void identity(Array<T> out)
{
    T *ptr = out.get();
    const dim4 out_dims  = out.dims();

    for (dim_t k = 0; k < out_dims[2] * out_dims[3]; k++) {
        for (dim_t j = 0; j < out_dims[1]; j++) {
            for (dim_t i = 0; i < out_dims[0]; i++) {
                ptr[j * out_dims[0] + i]  = (i == j) ? scalar<T>(1) : scalar<T>(0);
            }
        }
        ptr += out_dims[0] * out_dims[1];
    }
}

}
}
