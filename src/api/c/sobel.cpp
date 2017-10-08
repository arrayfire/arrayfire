/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <sobel.hpp>
#include <utility>

using af::dim4;
using namespace detail;

typedef std::pair<af_array, af_array> ArrayPair;
template<typename Ti, typename To>
ArrayPair sobelDerivatives(const af_array &in, const unsigned &ker_size)
{
    typedef std::pair< Array<To>, Array<To> > BAPair;
    BAPair  out = sobelDerivatives<Ti, To>(getArray<Ti>(in), ker_size);
    return std::make_pair(getHandle<To>(out.first),
                          getHandle<To>(out.second));
}

af_err af_sobel_operator(af_array *dx, af_array *dy, const af_array img, const unsigned ker_size)
{
    try {
        //FIXME: ADD SUPPORT FOR OTHER KERNEL SIZES
        //ARG_ASSERT(4, (ker_size==3 || ker_size==5 || ker_size==7));
        ARG_ASSERT(4, (ker_size==3));

        const ArrayInfo& info = getInfo(img);
        af::dim4 dims  = info.dims();

        DIM_ASSERT(3, (dims.ndims() >= 2));

        ArrayPair output;
        af_dtype type  = info.getType();
        switch(type) {
            case f32: output = sobelDerivatives<float , float> (img, ker_size); break;
            case f64: output = sobelDerivatives<double, double>(img, ker_size); break;
            case s32: output = sobelDerivatives<int   , int>   (img, ker_size); break;
            case u32: output = sobelDerivatives<uint  , int>   (img, ker_size); break;
            case s16: output = sobelDerivatives<short , int>   (img, ker_size); break;
            case u16: output = sobelDerivatives<ushort, int>   (img, ker_size); break;
            case b8 : output = sobelDerivatives<char  , int>   (img, ker_size); break;
            case u8:  output = sobelDerivatives<uchar , int>   (img, ker_size); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*dx, output.first);
        std::swap(*dy, output.second);
    }
    CATCHALL;

    return AF_SUCCESS;
}
