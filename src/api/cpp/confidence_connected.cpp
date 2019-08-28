/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af {

array confidenceCC(const array& in, const array& seedx, const array& seedy,
                   const unsigned radius, const unsigned multiplier,
                   const int iter, const double segmentedValue) {
    af_array temp = 0;
    AF_THROW(af_confidence_cc(&temp, in.get(), seedx.get(), seedy.get(),
                              radius, multiplier, iter, segmentedValue));
    return array(temp);
}

} // namespace af
