/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Array.hpp>

namespace af { class dim4; }

namespace cpu
{

    template<typename T>
    void copyData(T *data, const Array<T> &A);

    template<typename T>
    Array<T> copyArray(const Array<T> &A);

    template<typename inType, typename outType>
    void copyArray(Array<outType> &out, const Array<inType> &in);

    template<typename inType, typename outType>
    Array<outType> padArray(Array<inType> const &in, dim4 const &dims,
                            outType default_value=outType(0), double factor=1.0);

    template<typename T>
    void multiply_inplace(Array<T> &in, double val);

    template<typename T>
    T getScalar(const Array<T> &in);
}
