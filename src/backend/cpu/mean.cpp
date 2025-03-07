/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/mean.hpp>
#include <mean.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <complex>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<typename Ti, typename Tw, typename To>
using mean_dim_func = std::function<void(
    Param<To>, const dim_t, const CParam<Ti>, const dim_t, const int)>;

template<typename Ti, typename Tw, typename To>
Array<To> mean(const Array<Ti> &in, const int dim) {
    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    static const mean_dim_func<Ti, Tw, To> mean_funcs[] = {
        kernel::mean_dim<Ti, Tw, To, 1>(), kernel::mean_dim<Ti, Tw, To, 2>(),
        kernel::mean_dim<Ti, Tw, To, 3>(), kernel::mean_dim<Ti, Tw, To, 4>()};

    getQueue().enqueue(mean_funcs[in.ndims() - 1], out, 0, in, 0, dim);
    return out;
}

template<typename T, typename Tw>
using mean_weighted_dim_func =
    std::function<void(Param<T>, const dim_t, const CParam<T>, const dim_t,
                       const CParam<Tw>, const dim_t, const int)>;

template<typename T, typename Tw>
Array<T> mean(const Array<T> &in, const Array<Tw> &wt, const int dim) {
    dim4 odims   = in.dims();
    odims[dim]   = 1;
    Array<T> out = createEmptyArray<T>(odims);
    static const mean_weighted_dim_func<T, Tw> mean_funcs[] = {
        kernel::mean_weighted_dim<T, Tw, 1>(),
        kernel::mean_weighted_dim<T, Tw, 2>(),
        kernel::mean_weighted_dim<T, Tw, 3>(),
        kernel::mean_weighted_dim<T, Tw, 4>()};

    getQueue().enqueue(mean_funcs[in.ndims() - 1], out, 0, in, 0, wt, 0, dim);
    return out;
}

template<typename T, typename Tw>
T mean(const Array<T> &in, const Array<Tw> &wt) {
    using MeanOpT = kernel::MeanOp<compute_t<T>, compute_t<T>, compute_t<Tw>>;
    in.eval();
    wt.eval();
    getQueue().sync();

    af::dim4 dims    = in.dims();
    af::dim4 strides = in.strides();
    const T *inPtr   = in.get();
    const Tw *wtPtr  = wt.get();

    auto input  = compute_t<T>(inPtr[0]);
    auto weight = compute_t<Tw>(wtPtr[0]);
    MeanOpT Op(input, weight);

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;
                    Op(compute_t<T>(inPtr[idx]), compute_t<Tw>(wtPtr[idx]));
                }
            }
        }
    }

    return T(Op.runningMean);
}

template<typename Ti, typename Tw, typename To>
To mean(const Array<Ti> &in) {
    using MeanOpT = kernel::MeanOp<compute_t<Ti>, compute_t<To>, compute_t<Tw>>;
    in.eval();
    getQueue().sync();

    af::dim4 dims    = in.dims();
    af::dim4 strides = in.strides();
    const Ti *inPtr  = in.get();

    MeanOpT Op(0, 0);

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;
                    Op(compute_t<Ti>(inPtr[idx]), 1);
                }
            }
        }
    }

    return To(Op.runningMean);
}

#define INSTANTIATE(Ti, Tw, To)                        \
    template To mean<Ti, Tw, To>(const Array<Ti> &in); \
    template Array<To> mean<Ti, Tw, To>(const Array<Ti> &in, const int dim);

INSTANTIATE(double, double, double);
INSTANTIATE(float, float, float);
INSTANTIATE(int, float, float);
INSTANTIATE(unsigned, float, float);
INSTANTIATE(intl, double, double);
INSTANTIATE(uintl, double, double);
INSTANTIATE(short, float, float);
INSTANTIATE(ushort, float, float);
INSTANTIATE(uchar, float, float);
INSTANTIATE(char, float, float);
INSTANTIATE(cfloat, float, cfloat);
INSTANTIATE(cdouble, double, cdouble);
INSTANTIATE(half, float, half);
INSTANTIATE(half, float, float);

#define INSTANTIATE_WGT(T, Tw)                                              \
    template T mean<T, Tw>(const Array<T> &in, const Array<Tw> &wts);       \
    template Array<T> mean<T, Tw>(const Array<T> &in, const Array<Tw> &wts, \
                                  const int dim);

INSTANTIATE_WGT(double, double);
INSTANTIATE_WGT(float, float);
INSTANTIATE_WGT(cfloat, float);
INSTANTIATE_WGT(cdouble, double);
INSTANTIATE_WGT(half, float);

}  // namespace cpu
}  // namespace arrayfire
