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

array dilate(const array& in, const array& mask)
{
    af_array out = 0;
    AF_THROW(af_dilate(&out, in.get(), mask.get()));
    return array(out);
}

array dilate3(const array& in, const array& mask)
{
    af_array out = 0;
    AF_THROW(af_dilate3(&out, in.get(), mask.get()));
    return array(out);
}

array erode(const array& in, const array& mask)
{
    af_array out = 0;
    AF_THROW(af_erode(&out, in.get(), mask.get()));
    return array(out);
}

array erode3(const array& in, const array& mask)
{
    af_array out = 0;
    AF_THROW(af_erode3(&out, in.get(), mask.get()));
    return array(out);
}

}
