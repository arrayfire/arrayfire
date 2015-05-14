/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array cov(const array& X, const array& Y, const bool isbiased)
{
    af_array temp = 0;
    AF_THROW(af_cov(&temp, X.get(), Y.get(), isbiased));
    return array(temp);
}

}
