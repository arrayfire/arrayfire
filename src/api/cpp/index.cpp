/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/index.h>
#include "error.hpp"

namespace af
{

array moddims(const array& in, const unsigned ndims, const dim_type * const dims)
{
    af_array out = 0;
    AF_THROW(af_moddims(&out, in.get(), ndims, dims));
    return array(out);
}

array moddims(const array& in, const dim4& dims)
{
    return af::moddims(in, dims.ndims(), dims.get());
}

array moddims(const array& in, dim_type d0, dim_type d1, dim_type d2, dim_type d3)
{
    dim_type dims[4] = {d0, d1, d2, d3};
    return af::moddims(in, 4, dims);
}

array join(const int dim, const array& first, const array& second)
{
    af_array out = 0;
    AF_THROW(af_join(&out, dim, first.get(), second.get()));
    return array(out);
}

array tile(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    af_array out = 0;
    AF_THROW(af_tile(&out, in.get(), x, y, z, w));
    return array(out);
}

array reorder(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    af_array out = 0;
    AF_THROW(af_reorder(&out, in.get(), x, y, z, w));
    return array(out);
}

array shift(const array& in, const int x, const int y, const int z, const int w)
{
    af_array out = 0;
    AF_THROW(af_shift(&out, in.get(), x, y, z, w));
    return array(out);
}

}
