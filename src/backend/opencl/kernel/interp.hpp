/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>

namespace opencl {
    namespace kernel {

        template<af_interp_type method>
        const char *getInterpName()
        {
            switch(method) {
            case AF_INTERP_NEAREST:   return "NEAREST";
            case AF_INTERP_LINEAR:    return "LINEAR";
            case AF_INTERP_BILINEAR:  return "BILINEAR";
            case AF_INTERP_CUBIC:     return "CUBIC";
            case AF_INTERP_BICUBIC:   return "BICUBIC";
            case AF_INTERP_LOWER:     return "LOWER";
            default:                  return "NONE";
            }
        }

    }
}
