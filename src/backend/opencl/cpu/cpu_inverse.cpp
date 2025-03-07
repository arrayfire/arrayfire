/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <cpu/cpu_helper.hpp>
#include <cpu/cpu_inverse.hpp>
#include <cpu/cpu_lu.hpp>

namespace arrayfire {
namespace opencl {
namespace cpu {

template<typename T>
using getri_func_def = int (*)(ORDER_TYPE, int, T *, int, const int *);

#define INV_FUNC_DEF(FUNC) \
    template<typename T>   \
    FUNC##_func_def<T> FUNC##_func();

#define INV_FUNC(FUNC, TYPE, PREFIX)            \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

INV_FUNC_DEF(getri)
INV_FUNC(getri, float, s)
INV_FUNC(getri, double, d)
INV_FUNC(getri, cfloat, c)
INV_FUNC(getri, cdouble, z)

template<typename T>
Array<T> inverse(const Array<T> &in) {
    int M = in.dims()[0];
    // int N = in.dims()[1];

    // This condition is already handled in opencl/inverse.cpp
    // if (M != N) {
    // Array<T> I = identity<T>(in.dims());
    // return solve(in, I);
    //}

    Array<T> A = copyArray<T>(in);

    Array<int> pivot = cpu::lu_inplace<T>(A, false);

    mapped_ptr<T> aPtr   = A.getMappedPtr();
    mapped_ptr<int> pPtr = pivot.getMappedPtr();

    getri_func<T>()(AF_LAPACK_COL_MAJOR, M, aPtr.get(), A.strides()[1],
                    pPtr.get());

    return A;
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace opencl
}  // namespace arrayfire
#endif  // WITH_LINEAR_ALGEBRA
