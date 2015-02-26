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
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, af_pad_type pad>
Array<T> medfilt(const Array<T> &in, dim_type w_len, dim_type w_wid)
{
    ARG_ASSERT(2, (w_len<=kernel::MAX_MEDFILTER_LEN));

    const dim4 dims     = in.dims();

    Array<T> out      = createEmptyArray<T>(dims);

    kernel::medfilt<T, pad>(out, in, w_len, w_wid);

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> medfilt<T, AF_ZERO     >(const Array<T> &in, dim_type w_len, dim_type w_wid); \
    template Array<T> medfilt<T, AF_SYMMETRIC>(const Array<T> &in, dim_type w_len, dim_type w_wid);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
