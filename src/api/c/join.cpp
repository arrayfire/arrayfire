/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/data.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <join.hpp>
#include <vector>

using af::dim4;
using namespace detail;

template<typename Tx, typename Ty>
static inline af_array join(const int dim, const af_array first, const af_array second)
{
    return getHandle(join<Tx, Ty>(dim, getArray<Tx>(first), getArray<Ty>(second)));
}

template<typename T>
static inline af_array join_many(const int dim, const unsigned n_arrays, const af_array *inputs)
{
    std::vector<Array<T>> inputs_;
    inputs_.reserve(n_arrays);

    for(int i = 0; i < (int)n_arrays; i++) {
        inputs_.push_back(getArray<T>(inputs[i]));
    }
    return getHandle(join<T>(dim, inputs_));
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
            case s64: output = join<intl   , intl   >(dim, first, second);  break;
            case u64: output = join<uintl  , uintl  >(dim, first, second);  break;
            case u8:  output = join<uchar  , uchar  >(dim, first, second);  break;
            default:  TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs)
{
    try {
        ARG_ASSERT(3, n_arrays > 1 && n_arrays <= 10);

        std::vector<ArrayInfo> info;
        info.reserve(n_arrays);
        std::vector<af::dim4> dims(n_arrays);
        for(int i = 0; i < (int)n_arrays; i++) {
            info.push_back(getInfo(inputs[i]));
            dims[i] = info[i].dims();
        }

        ARG_ASSERT(1, dim >= 0 && dim < 4);

        for(int i = 1; i < (int)n_arrays; i++) {
            ARG_ASSERT(3, info[0].getType() == info[i].getType());
            DIM_ASSERT(3, info[i].elements() > 0);
        }

        // All dimensions except join dimension must be equal
        // Compute output dims
        for(int i = 0; i < 4; i++) {
            if(i != dim) {
                for(int j = 1; j < (int)n_arrays; j++) {
                    DIM_ASSERT(3, dims[0][i] == dims[j][i]);
                }
            }
        }

        af_array output;

        switch(info[0].getType()) {
            case f32: output = join_many<float  >(dim, n_arrays, inputs);  break;
            case c32: output = join_many<cfloat >(dim, n_arrays, inputs);  break;
            case f64: output = join_many<double >(dim, n_arrays, inputs);  break;
            case c64: output = join_many<cdouble>(dim, n_arrays, inputs);  break;
            case b8:  output = join_many<char   >(dim, n_arrays, inputs);  break;
            case s32: output = join_many<int    >(dim, n_arrays, inputs);  break;
            case u32: output = join_many<uint   >(dim, n_arrays, inputs);  break;
            case u8:  output = join_many<uchar  >(dim, n_arrays, inputs);  break;
            default:  TYPE_ERROR(1, info[0].getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
