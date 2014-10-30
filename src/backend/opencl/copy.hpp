/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <Array.hpp>

namespace opencl
{

    template<typename T>
    void copyData(T *data, const Array<T> &A);

    template<typename T>
    Array<T>* copyArray(const Array<T> &A);

    template<typename inType, typename outType>
    void copy(Array<outType> &dst, const Array<inType> &src, outType default_value, double factor);

}
