/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <clFFT.h>
#include <common/FFTPlanCache.hpp>
#include <memory.hpp>

#include <cstdio>

namespace arrayfire {
namespace opencl {
typedef clfftPlanHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char *_clfftGetResultString(clfftStatus st);

SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout, clfftDim rank,
                    size_t *clLengths, size_t *istrides, size_t idist,
                    size_t *ostrides, size_t odist, clfftPrecision precision,
                    size_t batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    friend SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout,
                               clfftDim rank, size_t *clLengths,
                               size_t *istrides, size_t idist, size_t *ostrides,
                               size_t odist, clfftPrecision precision,
                               size_t batch);
};
}  // namespace opencl
}  // namespace arrayfire

#define CLFFT_CHECK(fn)                                          \
    do {                                                         \
        clfftStatus _clfft_st = fn;                              \
        if (_clfft_st != CLFFT_SUCCESS) {                        \
            opencl::signalMemoryCleanup();                       \
            _clfft_st = (fn);                                    \
        }                                                        \
        if (_clfft_st != CLFFT_SUCCESS) {                        \
            char clfft_st_msg[1024];                             \
            snprintf(clfft_st_msg, sizeof(clfft_st_msg),         \
                     "clFFT Error (%d): %s\n", (int)(_clfft_st), \
                     opencl::_clfftGetResultString(_clfft_st));  \
                                                                 \
            AF_ERROR(clfft_st_msg, AF_ERR_INTERNAL);             \
        }                                                        \
    } while (0)
