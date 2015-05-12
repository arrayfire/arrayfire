/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/blas.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array transpose(const array& in, const bool conjugate)
{
    af_array out = 0;
    AF_THROW(af_transpose(&out, in.get(), conjugate));
    return array(out);
}

void transposeInPlace(array& in, const bool conjugate)
{
    AF_THROW(af_transpose_inplace(in.get(), conjugate));
}

}
