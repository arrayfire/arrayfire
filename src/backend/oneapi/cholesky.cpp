/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>
#include <cholesky.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
//#include <triangle.hpp>

namespace arrayfire {
namespace oneapi {

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace oneapi
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
