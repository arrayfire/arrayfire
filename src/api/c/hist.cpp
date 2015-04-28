/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <af/graphics.h>
#include <ArrayInfo.hpp>
#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <fghistogram1d.hpp>
#include <reduce.hpp>
#include <histogram.hpp>
#include <handle.hpp>

using af::dim4;
using namespace detail;
using namespace graphics;

template<typename T>
fg::Histogram* setup_histogram(const af_array X, const unsigned int nbins, const double minval, const double maxval)
{
    Array<T> xIn = getArray<T>(X);
    ArrayInfo Xinfo = getInfo(X);
    af::dim4 X_dims = Xinfo.dims();

    ForgeManager& fgMngr = ForgeManager::getInstance();
    fg::Histogram* hist = fgMngr.getHistogram(X_dims.elements(), getGLType<T>());

    Array<uint> hist_out = histogram<T, uint>(xIn, nbins, minval, maxval);

    uint ymax = reduce_all<af_max_t, uint, uint>(hist_out);
    uint ymin = reduce_all<af_min_t, uint, uint>(hist_out);
    hist->setAxesLimits(maxval, minval, ymax, ymin);

    copy_histogram<T>(hist_out, nbins, minval, maxval, hist);

    return hist;
}

af_err af_draw_histogram1d(const af_array X, const unsigned int nbins, const double minval, const double maxval)
{
    try {
        ArrayInfo Xinfo = getInfo(X);
        af::dim4 X_dims = Xinfo.dims();
        af_dtype Xtype  = Xinfo.getType();
        // TODO: Add error checking
        //DIM_ASSERT(0, X_dims == Y_dims);
        //DIM_ASSERT(0, Xinfo.isVector());

        //DYPE_ASSERT(Xtype == Ytype);

        fg::makeWindowCurrent(ForgeManager::getWindow());
        fg::Histogram* hist = NULL;

        switch(Xtype) {
            case f32: hist = setup_histogram<float  >(X, nbins, minval, maxval); break;
            case s32: hist = setup_histogram<int    >(X, nbins, minval, maxval); break;
            case u32: hist = setup_histogram<uint   >(X, nbins, minval, maxval); break;
            case u8 : hist = setup_histogram<uchar  >(X, nbins, minval, maxval); break;
            default:  TYPE_ERROR(1, Xtype);
        }

        int size = nbins*4;
        fg::drawHistogram(ForgeManager::getWindow(), *hist, nbins, minval, maxval);

    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif

