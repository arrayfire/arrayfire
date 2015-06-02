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

namespace af
{

array setunique(const array &in, const bool is_sorted)
{
    return setUnique(in, is_sorted);
}

array setUnique(const array &in, const bool is_sorted)
{
    af_array out = 0;
    AF_THROW(af_set_unique(&out, in.get(), is_sorted));
    return array(out);
}

array setunion(const array &first, const array &second, const bool is_unique)
{
    return setUnion(first, second, is_unique);
}

array setUnion(const array &first, const array &second, const bool is_unique)
{
    af_array out = 0;
    AF_THROW(af_set_union(&out, first.get(), second.get(), is_unique));
    return array(out);
}

array setintersect(const array &first, const array &second, const bool is_unique)
{
    return setIntersect(first, second, is_unique);
}

array setIntersect(const array &first, const array &second, const bool is_unique)
{
    af_array out = 0;
    AF_THROW(af_set_intersect(&out, first.get(), second.get(), is_unique));
    return array(out);
}

}
