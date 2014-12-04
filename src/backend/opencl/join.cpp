/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <join.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename Tx, typename Ty>
    Array<Tx> *join(const int dim, const Array<Tx> &first, const Array<Ty> &second, const af::dim4 &odims)
    {
        Array<Tx> *out = createEmptyArray<Tx>(odims);
        return out;
    }

#define INSTANTIATE(Tx, Ty)                                                             \
    template Array<Tx>* join<Tx, Ty>(const int dim, const Array<Tx> &first,             \
                                     const Array<Ty> &second, const af::dim4 &odims);   \

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)
}
