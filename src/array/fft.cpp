/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include "error.hpp"

namespace af
{

array fft(const array& in, double normalize, dim_type odim0)
{
    af_array out = 0;
    AF_THROW(af_fft(&out, in.get(), normalize, odim0));
    return array(out);
}

array fft2(const array& in, double normalize, dim_type odim0, dim_type odim1)
{
    af_array out = 0;
    AF_THROW(af_fft2(&out, in.get(), normalize, odim0, odim1));
    return array(out);
}

array fft3(const array& in, double normalize, dim_type odim0, dim_type odim1, dim_type odim2)
{
    af_array out = 0;
    AF_THROW(af_fft3(&out, in.get(), normalize, odim0, odim1, odim2));
    return array(out);
}

array ifft(const array& in, double normalize, dim_type odim0)
{
    af_array out = 0;
    AF_THROW(af_ifft(&out, in.get(), normalize, odim0));
    return array(out);
}

array ifft2(const array& in, double normalize, dim_type odim0, dim_type odim1)
{
    af_array out = 0;
    AF_THROW(af_ifft2(&out, in.get(), normalize, odim0, odim1));
    return array(out);
}

array ifft3(const array& in, double normalize, dim_type odim0, dim_type odim1, dim_type odim2)
{
    af_array out = 0;
    AF_THROW(af_ifft3(&out, in.get(), normalize, odim0, odim1, odim2));
    return array(out);
}

array fft(const array& in, dim_type odim0)
{
    double normalize = 1.0;
    af_array out = 0;
    AF_THROW(af_fft(&out, in.get(), normalize, odim0));
    return array(out);
}

array fft2(const array& in, dim_type odim0, dim_type odim1)
{
    double normalize = 1.0;
    af_array out = 0;
    AF_THROW(af_fft2(&out, in.get(), normalize, odim0, odim1));
    return array(out);
}

array fft3(const array& in, dim_type odim0, dim_type odim1, dim_type odim2)
{
    double normalize = 1.0;
    af_array out = 0;
    AF_THROW(af_fft3(&out, in.get(), normalize, odim0, odim1, odim2));
    return array(out);
}

array ifft(const array& in, dim_type odim0)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    double normalize = dim0;
    af_array out = 0;
    AF_THROW(af_ifft(&out, in.get(), normalize, odim0));
    return array(out);
}

array ifft2(const array& in, dim_type odim0, dim_type odim1)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    dim_type dim1 = odim1==0 ? dims[1] : odim1;
    double normalize = dim0*dim1;
    af_array out = 0;
    AF_THROW(af_ifft2(&out, in.get(), normalize, odim0, odim1));
    return array(out);
}

array ifft3(const array& in, dim_type odim0, dim_type odim1, dim_type odim2)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    dim_type dim1 = odim1==0 ? dims[1] : odim1;
    dim_type dim2 = odim2==0 ? dims[2] : odim2;
    double normalize = dim0*dim1*dim2;
    af_array out = 0;
    AF_THROW(af_ifft3(&out, in.get(), normalize, odim0, odim1, odim2));
    return array(out);
}

}
