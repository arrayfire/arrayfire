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
array sort(const array &in, const unsigned dim, const bool isAscending) {
    af_array out = 0;
    AF_THROW(af_sort(&out, in.get(), dim, isAscending));
    return array(out);
}

void sort(array &out, array &indices, const array &in, const unsigned dim,
          const bool isAscending) {
    af_array out_, indices_;
    AF_THROW(af_sort_index(&out_, &indices_, in.get(), dim, isAscending));
    out     = array(out_);
    indices = array(indices_);
}

void sort(array &out_keys, array &out_values, const array &keys,
          const array &values, const unsigned dim, const bool isAscending) {
    af_array okeys, ovalues;
    AF_THROW(af_sort_by_key(&okeys, &ovalues, keys.get(), values.get(), dim,
                            isAscending));
    out_keys   = array(okeys);
    out_values = array(ovalues);
}
}  // namespace af
