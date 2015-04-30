/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/graphics.h>
#include "error.hpp"

namespace af
{
    void image(const array &in)
    {
        AF_THROW(af_draw_image(in.get()));
    }

    void plot(const array &X, const array &Y)
    {
        AF_THROW(af_draw_plot(X.get(), Y.get()));
    }

    void hist(const array &X, const double minval, const double maxval)
    {
        AF_THROW(af_draw_hist(X.get(), minval, maxval));
    }
}
