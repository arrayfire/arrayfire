/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/index.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <join.hpp>

using af::dim4;
using namespace detail;

template<typename Tx, typename Ty>
static inline af_array join(const int dim, const af_array first, const af_array second)
{
    return getHandle(join<Tx, Ty>(dim, getArray<Tx>(first), getArray<Ty>(second)));
}

template<typename T>
static inline af_array join(const int dim, const af_array first, const af_array second,
                            const af_array third)
{
    return getHandle(join<T>(dim, getArray<T>(first), getArray<T>(second), getArray<T>(third)));
}

template<typename T>
static inline af_array join(const int dim, const af_array first, const af_array second,
                            const af_array third, const af_array fourth)
{
    return getHandle(join<T>(dim, getArray<T>(first), getArray<T>(second),
                             getArray<T>(third), getArray<T>(fourth)
                            ));
}

af_err af_join(af_array *out, const int dim, const af_array first, const af_array second)
{
    try {
        ArrayInfo finfo = getInfo(first);
        ArrayInfo sinfo = getInfo(second);
        af::dim4  fdims = finfo.dims();
        af::dim4  sdims = sinfo.dims();

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(2, finfo.getType() == sinfo.getType());
        DIM_ASSERT(2, sinfo.elements() > 0);
        DIM_ASSERT(3, finfo.elements() > 0);

        // All dimensions except join dimension must be equal
        // Compute output dims
        for(int i = 0; i < 4; i++) {
            if(i != dim) DIM_ASSERT(2, fdims[i] == sdims[i]);
        }

        af_array output;

        switch(finfo.getType()) {
            case f32: output = join<float  , float  >(dim, first, second);  break;
            case c32: output = join<cfloat , cfloat >(dim, first, second);  break;
            case f64: output = join<double , double >(dim, first, second);  break;
            case c64: output = join<cdouble, cdouble>(dim, first, second);  break;
            case b8:  output = join<char   , char   >(dim, first, second);  break;
            case s32: output = join<int    , int    >(dim, first, second);  break;
            case u32: output = join<uint   , uint   >(dim, first, second);  break;
            case u8:  output = join<uchar  , uchar  >(dim, first, second);  break;
            default:  TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_join3(af_array *out, const int dim, const af_array first, const af_array second, const af_array third)
{
    try {
        ArrayInfo finfo = getInfo(first);
        ArrayInfo sinfo = getInfo(second);
        ArrayInfo tinfo = getInfo(third);
        af::dim4  fdims = finfo.dims();
        af::dim4  sdims = sinfo.dims();
        af::dim4  tdims = tinfo.dims();

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(3, finfo.getType() == sinfo.getType());
        ARG_ASSERT(4, finfo.getType() == tinfo.getType());
        DIM_ASSERT(2, sinfo.elements() > 0);
        DIM_ASSERT(3, finfo.elements() > 0);
        DIM_ASSERT(4, tinfo.elements() > 0);

        // All dimensions except join dimension must be equal
        // Compute output dims
        for(int i = 0; i < 4; i++) {
            if(i != dim) DIM_ASSERT(3, fdims[i] == sdims[i]);
            if(i != dim) DIM_ASSERT(4, fdims[i] == tdims[i]);
        }

        af_array output;

        switch(finfo.getType()) {
            case f32: output = join<float  >(dim, first, second, third);  break;
            case c32: output = join<cfloat >(dim, first, second, third);  break;
            case f64: output = join<double >(dim, first, second, third);  break;
            case c64: output = join<cdouble>(dim, first, second, third);  break;
            case b8:  output = join<char   >(dim, first, second, third);  break;
            case s32: output = join<int    >(dim, first, second, third);  break;
            case u32: output = join<uint   >(dim, first, second, third);  break;
            case u8:  output = join<uchar  >(dim, first, second, third);  break;
            default:  TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_join4(af_array *out, const int dim, const af_array first, const af_array second,
                const af_array third, const af_array fourth)
{
    try {
        ArrayInfo finfo = getInfo(first);
        ArrayInfo sinfo = getInfo(second);
        ArrayInfo tinfo = getInfo(third);
        ArrayInfo rinfo = getInfo(fourth);
        af::dim4  fdims = finfo.dims();
        af::dim4  sdims = sinfo.dims();
        af::dim4  tdims = tinfo.dims();
        af::dim4  rdims = rinfo.dims();

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(3, finfo.getType() == sinfo.getType());
        ARG_ASSERT(4, finfo.getType() == tinfo.getType());
        ARG_ASSERT(5, finfo.getType() == rinfo.getType());
        DIM_ASSERT(2, sinfo.elements() > 0);
        DIM_ASSERT(3, finfo.elements() > 0);
        DIM_ASSERT(4, tinfo.elements() > 0);
        DIM_ASSERT(5, rinfo.elements() > 0);

        // All dimensions except join dimension must be equal
        // Compute output dims
        for(int i = 0; i < 4; i++) {
            if(i != dim) DIM_ASSERT(3, fdims[i] == sdims[i]);
            if(i != dim) DIM_ASSERT(4, fdims[i] == tdims[i]);
            if(i != dim) DIM_ASSERT(5, fdims[i] == rdims[i]);
        }

        af_array output;

        switch(finfo.getType()) {
            case f32: output = join<float  >(dim, first, second, third, fourth);  break;
            case c32: output = join<cfloat >(dim, first, second, third, fourth);  break;
            case f64: output = join<double >(dim, first, second, third, fourth);  break;
            case c64: output = join<cdouble>(dim, first, second, third, fourth);  break;
            case b8:  output = join<char   >(dim, first, second, third, fourth);  break;
            case s32: output = join<int    >(dim, first, second, third, fourth);  break;
            case u32: output = join<uint   >(dim, first, second, third, fourth);  break;
            case u8:  output = join<uchar  >(dim, first, second, third, fourth);  break;
            default:  TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
