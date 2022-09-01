/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_oneapi.hpp>
#include <lu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <blas.hpp>
#include <copy.hpp>
#include <platform.hpp>

namespace oneapi {

Array<int> convertPivot(int *ipiv, int in_sz, int out_sz) {
    ONEAPI_NOT_SUPPORTED("");
    Array<int> out = createEmptyArray<in_t>(af::dim4(1));
    return out;
}

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    ONEAPI_NOT_SUPPORTED("");
    Array<int> out = createEmptyArray<in_t>(af::dim4(1));
    return out;
}

bool isLAPACKAvailable() { return true; }

#define INSTANTIATE_LU(T)                                        \
    template Array<int> lu_inplace<T>(Array<T> & in,             \
                                      const bool convert_pivot); \
    template void lu<T>(Array<T> & lower, Array<T> & upper,      \
                        Array<int> & pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}  // namespace oneapi

#else  // WITH_LINEAR_ALGEBRA

namespace oneapi {

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
    AF_ERROR("Linear Algebra is disabled on OneAPI backend", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    ONEAPI_NOT_SUPPORTED("");
    AF_ERROR("Linear Algebra is disabled on OneAPI backend", AF_ERR_NOT_CONFIGURED);
}

bool isLAPACKAvailable() { return false; }

#define INSTANTIATE_LU(T)                                        \
    template Array<int> lu_inplace<T>(Array<T> & in,             \
                                      const bool convert_pivot); \
    template void lu<T>(Array<T> & lower, Array<T> & upper,      \
                        Array<int> & pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}  // namespace oneapi

#endif  // WITH_LINEAR_ALGEBRA
