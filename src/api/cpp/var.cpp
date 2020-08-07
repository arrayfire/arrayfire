/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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

array var(const array& in, const bool isbiased, const dim_t dim) {
    const af_var_bias bias =
        (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);
    return var(in, bias, dim);
}

array var(const array& in, const af_var_bias bias, const dim_t dim) {
    af_array temp = 0;
    AF_THROW(af_var_v2(&temp, in.get(), bias, getFNSD(dim, in.dims())));
    return array(temp);
}

array var(const array& in, const array& weights, const dim_t dim) {
    af_array temp = 0;
    AF_THROW(af_var_weighted(&temp, in.get(), weights.get(),
                             getFNSD(dim, in.dims())));
    return array(temp);
}

#define INSTANTIATE_VAR(T)                                                 \
    template<>                                                             \
    AFAPI T var(const array& in, const af_var_bias bias) {                 \
        double ret_val;                                                    \
        AF_THROW(af_var_all_v2(&ret_val, NULL, in.get(), bias));           \
        return cast<T>(ret_val);                                           \
    }                                                                      \
    template<>                                                             \
    AFAPI T var(const array& in, const bool isbiased) {                    \
        const af_var_bias bias =                                           \
            (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);      \
        return var<T>(in, bias);                                           \
    }                                                                      \
                                                                           \
    template<>                                                             \
    AFAPI T var(const array& in, const array& weights) {                   \
        double ret_val;                                                    \
        AF_THROW(                                                          \
            af_var_all_weighted(&ret_val, NULL, in.get(), weights.get())); \
        return cast<T>(ret_val);                                           \
    }

template<>
AFAPI af_cfloat var(const array& in, const af_var_bias bias) {
    double real, imag;
    AF_THROW(af_var_all_v2(&real, &imag, in.get(), bias));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
AFAPI af_cdouble var(const array& in, const af_var_bias bias) {
    double real, imag;
    AF_THROW(af_var_all_v2(&real, &imag, in.get(), bias));
    return {real, imag};
}

template<>
AFAPI af_cfloat var(const array& in, const bool isbiased) {
    const af_var_bias bias =
        (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);
    return var<af_cfloat>(in, bias);
}

template<>
AFAPI af_cdouble var(const array& in, const bool isbiased) {
    const af_var_bias bias =
        (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);
    return var<af_cdouble>(in, bias);
}

template<>
AFAPI af_cfloat var(const array& in, const array& weights) {
    double real, imag;
    AF_THROW(af_var_all_weighted(&real, &imag, in.get(), weights.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
AFAPI af_cdouble var(const array& in, const array& weights) {
    double real, imag;
    AF_THROW(af_var_all_weighted(&real, &imag, in.get(), weights.get()));
    return {real, imag};
}

INSTANTIATE_VAR(float);
INSTANTIATE_VAR(double);
INSTANTIATE_VAR(int);
INSTANTIATE_VAR(unsigned int);
INSTANTIATE_VAR(long long);
INSTANTIATE_VAR(unsigned long long);
INSTANTIATE_VAR(short);
INSTANTIATE_VAR(unsigned short);
INSTANTIATE_VAR(char);
INSTANTIATE_VAR(unsigned char);
INSTANTIATE_VAR(af_half);
INSTANTIATE_VAR(half_float::half);
#ifdef AF_CUDA
INSTANTIATE_VAR(__half);
#endif

#undef INSTANTIATE_VAR

}  // namespace af
