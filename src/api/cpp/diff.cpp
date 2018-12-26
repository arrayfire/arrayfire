/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/algorithm.h>
#include <af/array.h>
#include "error.hpp"

namespace af {
array diff1(const array &in, const int dim) {
    af_array out = 0;
    AF_THROW(af_diff1(&out, in.get(), dim));
    return array(out);
}

array diff2(const array &in, const int dim) {
    af_array out = 0;
    AF_THROW(af_diff2(&out, in.get(), dim));
    return array(out);
}
}  // namespace af
