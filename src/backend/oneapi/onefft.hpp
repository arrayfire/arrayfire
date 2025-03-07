/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/FFTPlanCache.hpp>
#include <memory.hpp>
#include <oneapi/mkl/dfti.hpp>

#include <cstdint>

namespace arrayfire {
namespace oneapi {

using ::oneapi::mkl::dft::domain;
using ::oneapi::mkl::dft::precision;

using PlanType   = std::shared_ptr<void>;
using SharedPlan = std::shared_ptr<PlanType>;

template<precision p, domain d>
PlanType findPlan(int rank, const bool isInPlace, int *n,
                  std::int64_t *istrides, int ibatch, std::int64_t *ostrides,
                  int obatch, int nbatch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    template<precision p, domain d>
    friend PlanType findPlan(int rank, const bool isInPlace, int *n,
                             std::int64_t *istrides, int ibatch,
                             std::int64_t *ostrides, int obatch, int nbatch);
};

}  // namespace oneapi
}  // namespace arrayfire
