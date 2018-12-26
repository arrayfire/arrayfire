/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/statistics.h>
#include "common.hpp"
#include "error.hpp"

namespace af {
void topk(array &values, array &indices, const array &in, const int k,
          const int dim, const topkFunction order) {
    af_array af_vals = 0;
    af_array af_idxs = 0;

    AF_THROW(af_topk(&af_vals, &af_idxs, in.get(), k, dim, order));

    values  = array(af_vals);
    indices = array(af_idxs);
}
}  // namespace af
