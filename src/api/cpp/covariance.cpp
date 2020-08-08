/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/statistics.h>
#include "error.hpp"

namespace af {

array cov(const array& X, const array& Y, const bool isbiased) {
    const af_var_bias bias =
        (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION);
    return cov(X, Y, bias);
}

array cov(const array& X, const array& Y, const af_var_bias bias) {
    af_array temp = 0;
    AF_THROW(af_cov_v2(&temp, X.get(), Y.get(), bias));
    return array(temp);
}

}  // namespace af
