/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/algorithm.h>
#include <af/compatible.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array histogram(const array &in, const unsigned nbins, const double minval, const double maxval)
{
    af_array out = 0;
    AF_THROW(af_histogram(&out, in.get(), nbins, minval, maxval));
    return array(out);
}

array histogram(const array &in, const unsigned nbins)
{
    af_array out = 0;
    AF_THROW(af_histogram(&out, in.get(), nbins, min<double>(in), max<double>(in)));
    return array(out);
}

array histequal(const array& in, const array& hist) { return histEqual(in, hist); }
array histEqual(const array& in, const array& hist)
{
    af_array temp = 0;
    AF_THROW(af_hist_equal(&temp, in.get(), hist.get()));
    return array(temp);
}

}
