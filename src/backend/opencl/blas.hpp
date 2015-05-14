/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/blas.h>
#include <Array.hpp>
#include <clBLAS.h>
#include <mutex>

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
    static std::once_flag clblasSetupFlag;
    call_once(clblasSetupFlag, clblasSetup);
}
}
