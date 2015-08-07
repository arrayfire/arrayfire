/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include <af/array.h>
#include <af/dim4.hpp>
#include "error.hpp"

namespace af
{


array fftNorm(const array& in, const double norm_factor, const dim_t odim0)
{
    af_array out = 0;
    AF_THROW(af_fft(&out, in.get(), norm_factor, odim0));
    return array(out);
}

array fft2Norm(const array& in, const double norm_factor, const dim_t odim0, const dim_t odim1)
{
    af_array out = 0;
    AF_THROW(af_fft2(&out, in.get(), norm_factor, odim0, odim1));
    return array(out);
}

array fft3Norm(const array& in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    af_array out = 0;
    AF_THROW(af_fft3(&out, in.get(), norm_factor, odim0, odim1, odim2));
    return array(out);
}

array fft(const array& in, const dim_t odim0)
{
    return fftNorm(in, 1.0, odim0);
}

array fft2(const array& in, const dim_t odim0, const dim_t odim1)
{
    return fft2Norm(in, 1.0, odim0, odim1);
}

array fft3(const array& in, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    return fft3Norm(in, 1.0, odim0, odim1, odim2);
}

array dft(const array& in, const double norm_factor, const dim4 outDims)
{
    array temp;
    switch(in.dims().ndims()) {
        case 1: temp = fftNorm(in, norm_factor, outDims[0]); break;
        case 2: temp = fft2Norm(in, norm_factor, outDims[0], outDims[1]); break;
        case 3: temp = fft3Norm(in, norm_factor, outDims[0], outDims[1], outDims[2]); break;
        default: AF_THROW(AF_ERR_NOT_SUPPORTED);
    }
    return temp;
}

array dft(const array& in, const dim4 outDims)
{
    return dft(in, 1.0, outDims);
}

array dft(const array& in)
{
    return dft(in, 1.0, dim4(0,0,0,0));
}

array ifftNorm(const array& in, const double norm_factor, const dim_t odim0)
{
    af_array out = 0;
    AF_THROW(af_ifft(&out, in.get(), norm_factor, odim0));
    return array(out);
}

array ifft2Norm(const array& in, const double norm_factor, const dim_t odim0, const dim_t odim1)
{
    af_array out = 0;
    AF_THROW(af_ifft2(&out, in.get(), norm_factor, odim0, odim1));
    return array(out);
}

array ifft3Norm(const array& in, const double norm_factor, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    af_array out = 0;
    AF_THROW(af_ifft3(&out, in.get(), norm_factor, odim0, odim1, odim2));
    return array(out);
}

array ifft(const array& in, const dim_t odim0)
{
    const dim4 dims = in.dims();
    dim_t dim0 = odim0==0 ? dims[0] : odim0;
    double norm_factor = 1.0/dim0;
    return ifftNorm(in, norm_factor, odim0);
}

array ifft2(const array& in, const dim_t odim0, const dim_t odim1)
{
    const dim4 dims = in.dims();
    dim_t dim0 = odim0==0 ? dims[0] : odim0;
    dim_t dim1 = odim1==0 ? dims[1] : odim1;
    double norm_factor = 1.0/(dim0*dim1);
    return ifft2Norm(in, norm_factor, odim0, odim1);
}

array ifft3(const array& in, const dim_t odim0, const dim_t odim1, const dim_t odim2)
{
    const dim4 dims = in.dims();
    dim_t dim0 = odim0==0 ? dims[0] : odim0;
    dim_t dim1 = odim1==0 ? dims[1] : odim1;
    dim_t dim2 = odim2==0 ? dims[2] : odim2;
    double norm_factor = 1.0/(dim0*dim1*dim2);
    return ifft3Norm(in, norm_factor, odim0, odim1, odim2);
}

array idft(const array& in, const double norm_factor, const dim4 outDims)
{
    array temp;
    switch(in.dims().ndims()) {
        case 1: temp =  ifftNorm(in, norm_factor, outDims[0]); break;
        case 2: temp = ifft2Norm(in, norm_factor, outDims[0], outDims[1]); break;
        case 3: temp = ifft3Norm(in, norm_factor, outDims[0], outDims[1], outDims[2]); break;
        default: AF_THROW(AF_ERR_NOT_SUPPORTED);
    }
    return temp;
}

array idft(const array& in, const dim4 outDims)
{
    return idft(in, 1.0, outDims);
}

array idft(const array& in)
{
    return idft(in, 1.0, dim4(0,0,0,0));
}

void fftInPlace(array& in, const double norm_factor)
{
    AF_THROW(af_fft_inplace(in.get(), norm_factor));
}

void fft2InPlace(array& in, const double norm_factor)
{
    AF_THROW(af_fft2_inplace(in.get(), norm_factor));
}

void fft3InPlace(array& in, const double norm_factor)
{
    AF_THROW(af_fft3_inplace(in.get(), norm_factor));
}

void ifftInPlace(array& in, const double norm_factor)
{
    const dim4 dims = in.dims();
    double norm = norm_factor *(1.0 / dims[0]);
    AF_THROW(af_ifft_inplace(in.get(), norm));
}

void ifft2InPlace(array& in, const double norm_factor)
{
    const dim4 dims = in.dims();
    double norm = norm_factor *(1.0 / (dims[0] * dims[1]));
    AF_THROW(af_ifft2_inplace(in.get(), norm));
}

void ifft3InPlace(array& in, const double norm_factor)
{
    const dim4 dims = in.dims();
    double norm = norm_factor *(1.0 / (dims[0] * dims[1] * dims[2]));
    AF_THROW(af_ifft3_inplace(in.get(), norm));
}

template<>
AFAPI array fftR2C<1>(const array &in, const dim4 &dims,
                      const double norm_factor)
{
    af_array res;
    AF_THROW(af_fft_r2c(&res, in.get(), norm_factor == 0 ? 1.0 : norm_factor, dims[0]));
    return array(res);
}

template<>
AFAPI array fftR2C<2>(const array &in, const dim4 &dims,
                      const double norm_factor)
{
    af_array res;
    AF_THROW(af_fft2_r2c(&res, in.get(), norm_factor == 0 ? 1.0 : norm_factor, dims[0], dims[1]));
    return array(res);
}

template<>
AFAPI array fftR2C<3>(const array &in, const dim4 &dims,
                      const double norm_factor)
{
    af_array res;
    AF_THROW(af_fft3_r2c(&res, in.get(), norm_factor == 0 ? 1.0 : norm_factor,
                         dims[0], dims[1], dims[2]));
    return array(res);
}

inline dim_t getOrigDim(dim_t d, bool is_odd)
{
    return 2 * (d - 1) + (is_odd ? 1 : 0);
}

template<>
AFAPI array fftC2R<1>(const array &in, const bool is_odd,
                      const double norm_factor)
{
    double norm = norm_factor;

    if (norm == 0) {
        dim4 idims = in.dims();
        dim_t dim0 = getOrigDim(idims[0], is_odd);
        norm = 1.0/dim0;
    }

    af_array res;
    AF_THROW(af_fft_c2r(&res, in.get(), norm, is_odd));
    return array(res);
}

template<>
AFAPI array fftC2R<2>(const array &in, const bool is_odd,
                      const double norm_factor)
{
    double norm = norm_factor;

    if (norm == 0) {
        dim4 idims = in.dims();
        dim_t dim0 = getOrigDim(idims[0], is_odd);
        dim_t dim1 = idims[1];
        norm = 1.0/(dim0 * dim1);
    }


    af_array res;
    AF_THROW(af_fft2_c2r(&res, in.get(), norm, is_odd));
    return array(res);
}

template<>
AFAPI array fftC2R<3>(const array &in, const bool is_odd,
                      const double norm_factor)
{
    double norm = norm_factor;

    if (norm == 0) {
        dim4 idims = in.dims();
        dim_t dim0 = getOrigDim(idims[0], is_odd);
        dim_t dim1 = idims[1];
        dim_t dim2 = idims[2];
        norm = 1.0/(dim0 * dim1 * dim2);
    }

    af_array res;
    AF_THROW(af_fft3_c2r(&res, in.get(), norm, is_odd));
    return array(res);
}

#define FFT_REAL(rank)                                      \
    template<>                                              \
    AFAPI array fftR2C<rank>(const array &in,               \
                             const double norm_factor)      \
    {                                                       \
        return fftR2C<rank>(in, in.dims(), norm_factor);    \
    }                                                       \
                                                            \

FFT_REAL(1)
FFT_REAL(2)
FFT_REAL(3)

}
