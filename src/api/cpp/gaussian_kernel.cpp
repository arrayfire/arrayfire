/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/array.h>
#include <af/algorithm.h>
#include <af/compatible.h>
#include "error.hpp"

namespace af{
    array gaussianKernel(const int rows, const int cols,
                         const double sig_r, const double sig_c)
    {
        af_array res;
        AF_THROW(af_gaussian_kernel(&res, rows, cols, sig_r, sig_c));
        return array(res);
    }

    // Compatible function
    array gaussiankernel(const int rows, const int cols,
                         const double sig_r, const double sig_c)
    {
        return gaussianKernel(rows, cols, sig_r, sig_c);
    }

}
