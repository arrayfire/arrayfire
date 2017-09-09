/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>

#include <complex>
#include <mean.hpp>
#include <kernel/mean.hpp>
#include <err_opencl.hpp>

using std::swap;
using af::dim4;
namespace opencl
{
    template<typename Ti, typename Tw, typename To>
    To mean(const Array<Ti>& in)
    {
        return kernel::mean_all<Ti, Tw, To>(in);
    }

    template<typename T, typename Tw>
    T mean(const Array<T>& in, const Array<Tw>& wts)
    {
        return kernel::mean_all_weighted<T, Tw>(in, wts);
    }

    template<typename Ti, typename Tw, typename To>
    Array<To> mean(const Array<Ti>& in, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;
        Array<To> out = createEmptyArray<To>(odims);
        kernel::mean<Ti, Tw, To>(out, in, dim);
        return out;
    }

    template<typename T, typename Tw>
    Array<T> mean(const Array<T>& in, const Array<Tw>& wts, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;
        Array<T> out = createEmptyArray<T>(odims);
        kernel::mean_weighted<T, Tw, T>(out, in, wts, dim);
        return out;
    }

    #define INSTANTIATE(Ti, Tw, To)                                         \
        template To mean<Ti, Tw, To>(const Array<Ti> &in); \
        template Array<To> mean<Ti, Tw, To>(const Array<Ti> &in, const int dim); \

    INSTANTIATE(double  , double,  double);
    INSTANTIATE(float   , float ,  float );
    INSTANTIATE(int     , float ,  float );
    INSTANTIATE(unsigned, float ,  float );
    INSTANTIATE(intl    , double,  double);
    INSTANTIATE(uintl   , double,  double);
    INSTANTIATE(short   , float ,  float );
    INSTANTIATE(ushort  , float ,  float );
    INSTANTIATE(uchar   , float ,  float );
    INSTANTIATE(char    , float ,  float );
    INSTANTIATE(cfloat  , float ,  cfloat);
    INSTANTIATE(cdouble , double, cdouble);

    #define INSTANTIATE_WGT(T, Tw)                                         \
        template T mean<T, Tw>(const Array<T> &in, const Array<Tw> &wts); \
        template Array<T> mean<T, Tw>(const Array<T> &in, const Array<Tw> &wts, const int dim); \

    INSTANTIATE_WGT(double , double);
    INSTANTIATE_WGT(float  , float );
    INSTANTIATE_WGT(cfloat , float );
    INSTANTIATE_WGT(cdouble, double);

}
