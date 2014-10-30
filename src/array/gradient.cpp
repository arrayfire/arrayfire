/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <utility>
#include "error.hpp"

namespace af
{

std::pair<array, array> gradient(const array& in)
{
    af_array rows = 0;
    af_array cols = 0;
    AF_THROW(af_gradient(&rows, &cols, in.get()));
    return std::make_pair(rows, cols);
}

}
