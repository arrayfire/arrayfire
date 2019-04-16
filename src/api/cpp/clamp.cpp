/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/gfor.h>
#include "error.hpp"

namespace af {
array clamp(const array &in, const array &lo, const array &hi) {
    af_array out;
    AF_THROW(af_clamp(&out, in.get(), lo.get(), hi.get(), gforGet()));
    return array(out);
}

array clamp(const array &in, const array &lo, const double hi) {
    return clamp(in, lo, constant(hi, lo.dims(), lo.type()));
}

array clamp(const array &in, const double lo, const array &hi) {
    return clamp(in, constant(lo, hi.dims(), hi.type()), hi);
}

array clamp(const array &in, const double lo, const double hi) {
    return clamp(in, constant(lo, in.dims(), in.type()),
                 constant(hi, in.dims(), in.type()));
}
}  // namespace af
