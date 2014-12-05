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
static inline af_array join(const int dim, const af_array first, const af_array second, const af::dim4 odims)
{
    return getHandle(*join<Tx, Ty>(dim, getArray<Tx>(first), getArray<Ty>(second), odims));
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
        af::dim4 odims;
        for(int i = 0; i < 4; i++) {
            if(i != dim) DIM_ASSERT(2, fdims[i] == sdims[i]);

            if(i == dim) {
                odims[i] = fdims[i] + sdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        af_array output;

        switch(finfo.getType()) {
            case f32: output = join<float  , float  >(dim, first, second, odims);  break;
            case c32: output = join<cfloat , cfloat >(dim, first, second, odims);  break;
            case f64: output = join<double , double >(dim, first, second, odims);  break;
            case c64: output = join<cdouble, cdouble>(dim, first, second, odims);  break;
            case b8:  output = join<char   , char   >(dim, first, second, odims);  break;
            case s32: output = join<int    , int    >(dim, first, second, odims);  break;
            case u32: output = join<uint   , uint   >(dim, first, second, odims);  break;
            case u8:  output = join<uchar  , uchar  >(dim, first, second, odims);  break;
            default:  TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
