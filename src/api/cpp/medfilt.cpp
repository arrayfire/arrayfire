/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include "error.hpp"

namespace af
{

array medfilt(const array& in, dim_type wind_length, dim_type wind_width, padType edge_pad)
{
    af_array out = 0;
    AF_THROW(af_medfilt(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

}
