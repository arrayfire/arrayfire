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

array histogram(const array &in, const unsigned nbins, const double minval, const double maxval)
{
    af_array out = 0;
    AF_THROW(af_histogram(&out, in.get(), nbins, minval, maxval));
    return array(out);
}

}
