/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/statistics.h>
#include "error.hpp"

using af::array;

namespace af {
void meanvar(array& mean, array& var, const array& in, const array& weights,
             const af_var_bias bias, const dim_t dim) {
    af_array mean_ = mean.get();
    af_array var_  = var.get();
    AF_THROW(af_meanvar(&mean_, &var_, in.get(), weights.get(), bias, dim));
    mean.set(mean_);
    var.set(var_);
}
}  // namespace af
