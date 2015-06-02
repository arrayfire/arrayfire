/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array matchTemplate(const array &searchImg, const array &templateImg, const matchType mType)
{
    af_array out = 0;
    AF_THROW(af_match_template(&out, searchImg.get(), templateImg.get(), mType));
    return array(out);
}

}
