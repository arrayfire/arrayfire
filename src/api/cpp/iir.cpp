/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include <af/array.h>
#include "error.hpp"
#include <algorithm>

namespace af
{

array fir(const array& b, const array& x)
{
    af_array out = 0;
    AF_THROW(af_fir(&out, b.get(), x.get()));
    return array(out);
}

array iir(const array &b, const array& a, const array& x)
{
    af_array out = 0;
    AF_THROW(af_iir(&out, b.get(), a.get(), x.get()));
    return array(out);
}

}
