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
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <medfilt.hpp>
#include <kernel/medfilt.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T, af_pad_type pad>
Array<T> * medfilt(const Array<T> &in, dim_type w_len, dim_type w_wid)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    ARG_ASSERT(2, (w_len<=kernel::MAX_MEDFILTER_LEN));

    const dim4 dims     = in.dims();

    Array<T> * out      = createEmptyArray<T>(dims);

    switch(w_len) {
        case  3: kernel::medfilt<T, pad,  3,  3>(*out, in); break;
        case  5: kernel::medfilt<T, pad,  5,  5>(*out, in); break;
        case  7: kernel::medfilt<T, pad,  7,  7>(*out, in); break;
        case  9: kernel::medfilt<T, pad,  9,  9>(*out, in); break;
        case 11: kernel::medfilt<T, pad, 11, 11>(*out, in); break;
        case 13: kernel::medfilt<T, pad, 13, 13>(*out, in); break;
        case 15: kernel::medfilt<T, pad, 15, 15>(*out, in); break;
    }
    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * medfilt<T, AF_ZERO     >(const Array<T> &in, dim_type w_len, dim_type w_wid); \
    template Array<T> * medfilt<T, AF_SYMMETRIC>(const Array<T> &in, dim_type w_len, dim_type w_wid);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
