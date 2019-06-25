/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <internal_enums.hpp>
#include <af/defines.h>

#include <utility>

namespace cuda {

static __DH__ dim_t trimIndex(const int &idx, const dim_t &len) {
    int ret_val = idx;
    if (ret_val < 0) {
        int offset = (abs(ret_val) - 1) % len;
        ret_val    = offset;
    } else if (ret_val >= len) {
        int offset = abs(ret_val) % len;
        ret_val    = len - offset - 1;
    }
    return ret_val;
}

// TODO(pradeep) think of a better name
std::pair<InterpolationType, int> toInternalEnum(const af_interp_type p);

BorderType toInternalEnum(const af_border_type p);

MomentType toInternalEnum(const af_moment_type p);

FluxFunction toInternalEnum(const af_flux_function p);

ErrorMetric toInternalEnum(const af_match_type p);

}  // namespace cuda
