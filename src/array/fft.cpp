/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/fft.h>
#include "error.hpp"

namespace af
{

array fft(const array& in, double normalize, dim_type pad0)
{
    af_array out = 0;
    AF_THROW(af_fft(&out, in.get(), normalize, pad0));
    return array(out);
}

array fft2(const array& in, double normalize, dim_type pad0, dim_type pad1)
{
    af_array out = 0;
    AF_THROW(af_fft2(&out, in.get(), normalize, pad0, pad1));
    return array(out);
}

array fft3(const array& in, double normalize, dim_type pad0, dim_type pad1, dim_type pad2)
{
    af_array out = 0;
    AF_THROW(af_fft3(&out, in.get(), normalize, pad0, pad1, pad2));
    return array(out);
}

array ifft(const array& in, double normalize, dim_type pad0)
{
    af_array out = 0;
    AF_THROW(af_ifft(&out, in.get(), normalize, pad0));
    return array(out);
}

array ifft2(const array& in, double normalize, dim_type pad0, dim_type pad1)
{
    af_array out = 0;
    AF_THROW(af_ifft2(&out, in.get(), normalize, pad0, pad1));
    return array(out);
}

array ifft3(const array& in, double normalize, dim_type pad0, dim_type pad1, dim_type pad2)
{
    af_array out = 0;
    AF_THROW(af_ifft3(&out, in.get(), normalize, pad0, pad1, pad2));
    return array(out);
}

}
