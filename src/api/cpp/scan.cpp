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
array accum(const array& in, const int dim) {
    af_array out = 0;
    AF_THROW(af_accum(&out, in.get(), dim));
    return array(out);
}

array scan(const array& in, const int dim, binaryOp op, bool inclusive_scan) {
    af_array out = 0;
    AF_THROW(af_scan(&out, in.get(), dim, op, inclusive_scan));
    return array(out);
}

array scanByKey(const array& key, const array& in, const int dim, binaryOp op,
                bool inclusive_scan) {
    af_array out = 0;
    AF_THROW(
        af_scan_by_key(&out, key.get(), in.get(), dim, op, inclusive_scan));
    return array(out);
}
}  // namespace af
