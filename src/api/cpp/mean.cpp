/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/algorithm.h>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/statistics.h>
#include "common.hpp"
#include "error.hpp"
#include "half.hpp"
#ifdef AF_CUDA
#include <cuda_fp16.h>
#include <traits.hpp>
#endif

namespace af {

array mean(const array& in, const dim_t dim) {
    af_array temp = 0;
    AF_THROW(af_mean(&temp, in.get(), getFNSD(dim, in.dims())));
    return array(temp);
}

array mean(const array& in, const array& weights, const dim_t dim) {
    af_array temp = 0;
    AF_THROW(af_mean_weighted(&temp, in.get(), weights.get(),
                              getFNSD(dim, in.dims())));
    return array(temp);
}

#define INSTANTIATE_MEAN(T)                                                  \
    template<>                                                               \
    AFAPI T mean(const array& in) {                                          \
        double ret_val;                                                      \
        AF_THROW(af_mean_all(&ret_val, NULL, in.get()));                     \
        return cast<T>(ret_val);                                             \
    }                                                                        \
    template<>                                                               \
    AFAPI T mean(const array& in, const array& wts) {                        \
        double ret_val;                                                      \
        AF_THROW(af_mean_all_weighted(&ret_val, NULL, in.get(), wts.get())); \
        return cast<T>(ret_val);                                             \
    }

template<>
AFAPI af_cfloat mean(const array& in) {
    double real, imag;
    AF_THROW(af_mean_all(&real, &imag, in.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
AFAPI af_cdouble mean(const array& in) {
    double real, imag;
    AF_THROW(af_mean_all(&real, &imag, in.get()));
    return {real, imag};
}

template<>
AFAPI af_cfloat mean(const array& in, const array& weights) {
    double real, imag;
    AF_THROW(af_mean_all_weighted(&real, &imag, in.get(), weights.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
AFAPI af_cdouble mean(const array& in, const array& weights) {
    double real, imag;
    AF_THROW(af_mean_all_weighted(&real, &imag, in.get(), weights.get()));
    return {real, imag};
}

INSTANTIATE_MEAN(float);
INSTANTIATE_MEAN(double);
INSTANTIATE_MEAN(int);
INSTANTIATE_MEAN(unsigned int);
INSTANTIATE_MEAN(char);
INSTANTIATE_MEAN(unsigned char);
INSTANTIATE_MEAN(long long);
INSTANTIATE_MEAN(unsigned long long);
INSTANTIATE_MEAN(short);
INSTANTIATE_MEAN(unsigned short);
INSTANTIATE_MEAN(af_half);
INSTANTIATE_MEAN(half_float::half);  // Add support for public API
#ifdef AF_CUDA
INSTANTIATE_MEAN(__half);
#endif

#undef INSTANTIATE_MEAN

}  // namespace af
