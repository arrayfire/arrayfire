/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include "error.hpp"
#include <algorithm>

namespace af
{

array convolve(const array& signal, const array& filter, bool expand)
{
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch(std::max(sN,fN)) {
        case 1: return convolve1(signal, filter, expand);
        case 2: return convolve2(signal, filter, expand);
        case 3: return convolve3(signal, filter, expand);
        default: return convolve3(signal, filter, expand);
    }
}

array convolve(const array& signal, const array& col_filter, const array& row_filter, bool expand)
{
    af_array out = 0;
    AF_THROW(af_convolve2_sep(&out, signal.get(), col_filter.get(), row_filter.get(), expand));
    return array(out);
}

array convolve1(const array& signal, const array& filter, bool expand)
{
    af_array out = 0;
    AF_THROW(af_convolve1(&out, signal.get(), filter.get(), expand));
    return array(out);
}

array convolve2(const array& signal, const array& filter, bool expand)
{
    af_array out = 0;
    AF_THROW(af_convolve2(&out, signal.get(), filter.get(), expand));
    return array(out);
}

array convolve3(const array& signal, const array& filter, bool expand)
{
    af_array out = 0;
    AF_THROW(af_convolve3(&out, signal.get(), filter.get(), expand));
    return array(out);
}

}
