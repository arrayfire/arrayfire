/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/array.h>
#include <af/gfor.h>

namespace af
{

    bool gfor_toggle()
    {
        static bool gfor_flag;
        gfor_flag ^= 1;
        return gfor_flag;
    }

}
