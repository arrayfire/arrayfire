/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <mutex>

// TODO: Temporary choose between clBLAS and CLBlast here
#define USE_CLBLAS // or USE_CLBLAST

#if defined(USE_CLBLAS)
#include <clBLAS.h>
#elif defined(USE_CLBLAST)
#include <clblast.h>
#else
#error "Define either USE_CLBLAS or USE_CLBLAST"
#endif

namespace opencl
{

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs);
template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs);

STATIC_ void
initBlas() {
#if defined(USE_CLBLAS)
    static std::once_flag clblasSetupFlag;
    call_once(clblasSetupFlag, clblasSetup);
#endif // USE_CLBLAS
}
}
