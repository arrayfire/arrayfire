/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_common.hpp>
#include <solve.hpp>

#if defined(WITH_OPENCL_LINEAR_ALGEBRA)
#include <magma/magma.h>
#include <lu.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <blas.hpp>

namespace opencl
{

template<typename T>
Array<T> solve_square(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{

    dim4 iDims = a.dims();
    int M = iDims[0];
    int N = iDims[1];
    int MN = std::min(M, N);
    int *ipiv = new int[MN];

    Array<T> A = copyArray<T>(a);
    Array<T> B = copyArray<T>(b);

    cl::Buffer *a_buf = A.get();
    int info = 0;
    magma_getrf_gpu<T>(M, N, (*a_buf)(), a.getOffset(), a.strides()[1],
                       ipiv, getQueue()(), &info);

    cl::Buffer *b_buf = B.get();
    int K = b.dims()[1];
    magma_getrs_gpu<T>(MagmaNoTrans, M, K,
                       (*a_buf)(), a.getOffset(), a.strides()[1],
                       ipiv,
                       (*b_buf)(), b.getOffset(), b.strides()[1],
                       getQueue()(), &info);
    return B;
}


template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    try {
        initBlas();
        if(a.dims()[0] == a.dims()[1]) {
            return solve_square<T>(a, b, options);
        } else {
            AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
            //return solve_rect<T>(a, b, options);
        }
    } catch(cl::Error &err) {
        CL_TO_AF_ERROR(err);
    }
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#else

namespace opencl
{

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b, const af_solve_t options)
{
    AF_ERROR("Linear Algebra is disabled on OpenCL", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_SOLVE(T)                                                                   \
    template Array<T> solve<T> (const Array<T> &a, const Array<T> &b, const af_solve_t options);

INSTANTIATE_SOLVE(float)
INSTANTIATE_SOLVE(cfloat)
INSTANTIATE_SOLVE(double)
INSTANTIATE_SOLVE(cdouble)

}

#endif
