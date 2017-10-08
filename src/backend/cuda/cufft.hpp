/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/err_common.hpp>
#include <cufft.h>
#include <common/FFTPlanCache.hpp>
#include <cstdio>

namespace cuda
{
typedef cufftHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char * _cufftGetResultString(cufftResult res);

SharedPlan findPlan(int rank, int *n,
                    int *inembed, int istride, int idist,
                    int *onembed, int ostride, int odist,
                    cufftType type, int batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType>
{
    friend SharedPlan findPlan(int rank, int *n,
                               int *inembed, int istride, int idist,
                               int *onembed, int ostride, int odist,
                               cufftType type, int batch);
};
}

#define CUFFT_CHECK(fn) do {                        \
        cufftResult _cufft_res = fn;                \
        if (_cufft_res != CUFFT_SUCCESS) {          \
            char cufft_res_msg[1024];               \
            snprintf(cufft_res_msg,                 \
                     sizeof(cufft_res_msg),         \
                     "cuFFT Error (%d): %s\n",      \
                     (int)(_cufft_res),             \
                     cuda::_cufftGetResultString(   \
                         _cufft_res));              \
                                                    \
            AF_ERROR(cufft_res_msg,                 \
                     AF_ERR_INTERNAL);              \
        }                                           \
    } while(0)
