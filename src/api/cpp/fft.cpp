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


array fft(const array& in, double norm_factor, dim_type odim0)
{
    af_array out = 0;
    AF_THROW(af_fft(&out, in.get(), norm_factor, odim0));
    return array(out);
}

array fft2(const array& in, double norm_factor, dim_type odim0, dim_type odim1)
{
    af_array out = 0;
    AF_THROW(af_fft2(&out, in.get(), norm_factor, odim0, odim1));
    return array(out);
}

array fft3(const array& in, double norm_factor, dim_type odim0, dim_type odim1, dim_type odim2)
{
    af_array out = 0;
    AF_THROW(af_fft3(&out, in.get(), norm_factor, odim0, odim1, odim2));
    return array(out);
}

array fft(const array& in, dim_type odim0)
{
    return fft(in, 1.0, odim0);
}

array fft2(const array& in, dim_type odim0, dim_type odim1)
{
    return fft2(in, 1.0, odim0, odim1);
}

array fft3(const array& in, dim_type odim0, dim_type odim1, dim_type odim2)
{
    return fft3(in, 1.0, odim0, odim1, odim2);
}

array fft(const array& in, double norm_factor, const dim4 outDims)
{
    array temp;
    switch(in.dims().ndims()) {
        case 1: temp = fft(in, norm_factor, outDims[0]); break;
        case 2: temp = fft2(in, norm_factor, outDims[0], outDims[1]); break;
        case 3: temp = fft3(in, norm_factor, outDims[0], outDims[1], outDims[2]); break;
        default: AF_THROW(AF_ERR_NOT_SUPPORTED);
    }
    return temp;
}

array fft(const array& in, const dim4 outDims)
{
    return fft(in, 1.0, outDims);
}

array fft(const array& in)
{
    return fft(in, 1.0, dim4(0,0,0,0));
}

array ifft(const array& in, double norm_factor, dim_type odim0)
{
    af_array out = 0;
    AF_THROW(af_ifft(&out, in.get(), norm_factor, odim0));
    return array(out);
}

array ifft2(const array& in, double norm_factor, dim_type odim0, dim_type odim1)
{
    af_array out = 0;
    AF_THROW(af_ifft2(&out, in.get(), norm_factor, odim0, odim1));
    return array(out);
}

array ifft3(const array& in, double norm_factor, dim_type odim0, dim_type odim1, dim_type odim2)
{
    af_array out = 0;
    AF_THROW(af_ifft3(&out, in.get(), norm_factor, odim0, odim1, odim2));
    return array(out);
}

array ifft(const array& in, dim_type odim0)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    double norm_factor = 1.0/dim0;
    return ifft(in, norm_factor, odim0);
}

array ifft2(const array& in, dim_type odim0, dim_type odim1)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    dim_type dim1 = odim1==0 ? dims[1] : odim1;
    double norm_factor = 1.0/(dim0*dim1);
    return ifft2(in, norm_factor, odim0, odim1);
}

array ifft3(const array& in, dim_type odim0, dim_type odim1, dim_type odim2)
{
    const dim4 dims = in.dims();
    dim_type dim0 = odim0==0 ? dims[0] : odim0;
    dim_type dim1 = odim1==0 ? dims[1] : odim1;
    dim_type dim2 = odim2==0 ? dims[2] : odim2;
    double norm_factor = 1.0/(dim0*dim1*dim2);
    return ifft3(in, norm_factor, odim0, odim1, odim2);
}

array ifft(const array& in, double norm_factor, const dim4 outDims)
{
    array temp;
    switch(in.dims().ndims()) {
        case 1: temp =  ifft(in, norm_factor, outDims[0]); break;
        case 2: temp = ifft2(in, norm_factor, outDims[0], outDims[1]); break;
        case 3: temp = ifft3(in, norm_factor, outDims[0], outDims[1], outDims[2]); break;
        default: AF_THROW(AF_ERR_NOT_SUPPORTED);
    }
    return temp;
}

array ifft(const array& in, const dim4 outDims)
{
    return ifft(in, 1.0, outDims);
}

array ifft(const array& in)
{
    return ifft(in, 1.0, dim4(0,0,0,0));
}

}
