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
#include <kernel/join.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename Tx, typename Ty>
    Array<Tx> *join(const int dim, const Array<Tx> &first, const Array<Ty> &second)
    {
        if ((std::is_same<Tx, double>::value || std::is_same<Tx, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }
        if ((std::is_same<Ty, double>::value || std::is_same<Ty, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }

        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<Tx> *out = createEmptyArray<Tx>(odims);

        switch(dim) {
            case 0: kernel::join<Tx, Ty, 0>(*out, first, second);
                    break;
            case 1: kernel::join<Tx, Ty, 1>(*out, first, second);
                    break;
            case 2: kernel::join<Tx, Ty, 2>(*out, first, second);
                    break;
            case 3: kernel::join<Tx, Ty, 3>(*out, first, second);
                    break;
        }

        return out;
    }

#define INSTANTIATE(Tx, Ty)                                                                             \
    template Array<Tx>* join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);   \

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)
}
