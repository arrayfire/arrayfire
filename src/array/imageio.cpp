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

array loadImage(const char* filename, const bool is_color)
{
    af_array out = 0;
    AF_THROW(af_load_image(&out, filename, is_color));
    return array(out);
}

void saveImage(const char* filename, const array& in)
{
    AF_THROW(af_save_image(filename, in.get()));
}

}
