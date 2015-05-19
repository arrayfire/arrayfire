/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>

namespace af
{

/// Get the first non-zero dimension
static inline dim_t getFNSD(const int dim, af::dim4 dims)
{
    if(dim >= 0)
        return dim;

    dim_t fNSD = 0;
    for (dim_t i=0; i<4; ++i) {
        if (dims[i]>1) {
            fNSD = i;
            break;
        }
    }
    return fNSD;
}

}
