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

namespace af {

#define INSTANTIATE_STDEV(T)                                       \
    template<>                                                     \
    AFAPI T stdev(const array& in, const af_var_bias bias) {       \
        double ret_val;                                            \
        AF_THROW(af_stdev_all_v2(&ret_val, NULL, in.get(), bias)); \
        return (T)ret_val;                                         \
    }                                                              \
    template<>                                                     \
    AFAPI T stdev(const array& in) {                               \
        return stdev<T>(in, AF_VARIANCE_POPULATION);               \
    }

template<>
AFAPI af_cfloat stdev(const array& in, const af_var_bias bias) {
    double real, imag;
    AF_THROW(af_stdev_all_v2(&real, &imag, in.get(), bias));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
AFAPI af_cdouble stdev(const array& in, const af_var_bias bias) {
    double real, imag;
    AF_THROW(af_stdev_all_v2(&real, &imag, in.get(), bias));
    return {real, imag};
}

template<>
AFAPI af_cfloat stdev(const array& in) {
    return stdev<af_cfloat>(in, AF_VARIANCE_POPULATION);
}

template<>
AFAPI af_cdouble stdev(const array& in) {
    return stdev<af_cdouble>(in, AF_VARIANCE_POPULATION);
}

INSTANTIATE_STDEV(float);
INSTANTIATE_STDEV(double);
INSTANTIATE_STDEV(int);
INSTANTIATE_STDEV(unsigned int);
INSTANTIATE_STDEV(long long);
INSTANTIATE_STDEV(unsigned long long);
INSTANTIATE_STDEV(short);
INSTANTIATE_STDEV(unsigned short);
INSTANTIATE_STDEV(char);
INSTANTIATE_STDEV(unsigned char);

#undef INSTANTIATE_STDEV

array stdev(const array& in, const af_var_bias bias, const dim_t dim) {
    af_array temp = 0;
    AF_THROW(af_stdev_v2(&temp, in.get(), bias, getFNSD(dim, in.dims())));
    return array(temp);
}

array stdev(const array& in, const dim_t dim) {
    return stdev(in, AF_VARIANCE_POPULATION, dim);
}

}  // namespace af
