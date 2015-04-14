/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <af/array.h>
#include <af/graphics.h>
#include "error.hpp"

namespace af
{
    void drawImage(const array &in, const fg_image_handle image)
    {
        AF_THROW(af_draw_image(in.get(), image));
    }

    void drawPlot(const array &X,const array &Y, const fg_plot_handle plot)
    {
        AF_THROW(af_draw_plot(X.get(), Y.get(), plot));
    }

}

#endif
