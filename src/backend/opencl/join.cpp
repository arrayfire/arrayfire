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
    template<int dim>
    af::dim4 calcOffset(const af::dim4 dims)
    {
        af::dim4 offset;
        offset[0] = (dim == 0) ? dims[0] : 0;
        offset[1] = (dim == 1) ? dims[1] : 0;
        offset[2] = (dim == 2) ? dims[2] : 0;
        offset[3] = (dim == 3) ? dims[3] : 0;
        return offset;
    }

    template<typename Tx, typename Ty>
    Array<Tx> join(const int dim, const Array<Tx> &first, const Array<Ty> &second)
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

        Array<Tx> out = createEmptyArray<Tx>(odims);

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                kernel::join<Tx, Tx, 0>(out, first,  zero);
                kernel::join<Tx, Ty, 0>(out, second, calcOffset<0>(fdims));
                break;
            case 1:
                kernel::join<Tx, Tx, 1>(out, first,  zero);
                kernel::join<Tx, Ty, 1>(out, second, calcOffset<1>(fdims));
                break;
            case 2:
                kernel::join<Tx, Tx, 2>(out, first,  zero);
                kernel::join<Tx, Ty, 2>(out, second, calcOffset<2>(fdims));
                break;
            case 3:
                kernel::join<Tx, Tx, 3>(out, first,  zero);
                kernel::join<Tx, Ty, 3>(out, second, calcOffset<3>(fdims));
                break;
        }

        return out;
    }

    template<typename T>
    Array<T> join(const int dim, const Array<T> &first, const Array<T> &second, const Array<T> &third)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }

        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();
        af::dim4 tdims = third.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i] + tdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<T> out = createEmptyArray<T>(odims);

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                kernel::join<T, T, 0>(out, first,  zero);
                kernel::join<T, T, 0>(out, second, calcOffset<0>(fdims));
                kernel::join<T, T, 0>(out, third,  calcOffset<0>(fdims + sdims));
                break;
            case 1:
                kernel::join<T, T, 1>(out, first,  zero);
                kernel::join<T, T, 1>(out, second, calcOffset<1>(fdims));
                kernel::join<T, T, 1>(out, third,  calcOffset<1>(fdims + sdims));
                break;
            case 2:
                kernel::join<T, T, 2>(out, first,  zero);
                kernel::join<T, T, 2>(out, second, calcOffset<2>(fdims));
                kernel::join<T, T, 2>(out, third,  calcOffset<2>(fdims + sdims));
                break;
            case 3:
                kernel::join<T, T, 3>(out, first,  zero);
                kernel::join<T, T, 3>(out, second, calcOffset<3>(fdims));
                kernel::join<T, T, 3>(out, third,  calcOffset<3>(fdims + sdims));
                break;
        }
        return out;
    }

    template<typename T>
    Array<T> join(const int dim, const Array<T> &first, const Array<T> &second,
                  const Array<T> &third, const Array<T> &fourth)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }

        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();
        af::dim4 tdims = third.dims();
        af::dim4 rdims = fourth.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i] + tdims[i] + rdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<T> out = createEmptyArray<T>(odims);

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                kernel::join<T, T, 0>(out, first,  zero);
                kernel::join<T, T, 0>(out, second, calcOffset<0>(fdims));
                kernel::join<T, T, 0>(out, third,  calcOffset<0>(fdims + sdims));
                kernel::join<T, T, 0>(out, fourth, calcOffset<0>(fdims + sdims + rdims));
                break;
            case 1:
                kernel::join<T, T, 1>(out, first,  zero);
                kernel::join<T, T, 1>(out, second, calcOffset<1>(fdims));
                kernel::join<T, T, 1>(out, third,  calcOffset<1>(fdims + sdims));
                kernel::join<T, T, 0>(out, fourth, calcOffset<1>(fdims + sdims + rdims));
                break;
            case 2:
                kernel::join<T, T, 2>(out, first,  zero);
                kernel::join<T, T, 2>(out, second, calcOffset<2>(fdims));
                kernel::join<T, T, 2>(out, third,  calcOffset<2>(fdims + sdims));
                kernel::join<T, T, 0>(out, fourth, calcOffset<2>(fdims + sdims + rdims));
                break;
            case 3:
                kernel::join<T, T, 3>(out, first,  zero);
                kernel::join<T, T, 3>(out, second, calcOffset<3>(fdims));
                kernel::join<T, T, 3>(out, third,  calcOffset<3>(fdims + sdims));
                kernel::join<T, T, 3>(out, fourth, calcOffset<3>(fdims + sdims + rdims));
                break;
        }
        return out;
    }

#define INSTANTIATE(Tx, Ty)                                                                             \
    template Array<Tx> join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);   \

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)

#undef INSTANTIATE

#define INSTANTIATE(T)                                                                              \
    template Array<T> join<T>(const int dim, const Array<T> &first, const Array<T> &second,         \
                              const Array<T> &third);                                               \
    template Array<T> join<T>(const int dim, const Array<T> &first, const Array<T> &second,         \
                              const Array<T> &third, const Array<T> &fourth);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#undef INSTANTIATE
}
