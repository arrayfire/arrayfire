/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/arith.h>
#include "error.hpp"

namespace af
{
#define INSTANTIATE(func)                                           \
    array func(const array &in)                                     \
    {                                                               \
        af_array out = 0;                                           \
        af_##func(&out, in.get());                                  \
        return array(out);                                          \
    }

    INSTANTIATE(cplx  )
    INSTANTIATE(abs   )

    INSTANTIATE(round )
    INSTANTIATE(floor )
    INSTANTIATE(ceil  )

    INSTANTIATE(sin   )
    INSTANTIATE(cos   )
    INSTANTIATE(tan   )

    INSTANTIATE(asin  )
    INSTANTIATE(acos  )
    INSTANTIATE(atan  )

    INSTANTIATE(sinh  )
    INSTANTIATE(cosh  )
    INSTANTIATE(tanh  )

    INSTANTIATE(asinh )
    INSTANTIATE(acosh )
    INSTANTIATE(atanh )

    INSTANTIATE(exp   )
    INSTANTIATE(expm1 )
    INSTANTIATE(erf   )
    INSTANTIATE(erfc  )

    INSTANTIATE(log   )
    INSTANTIATE(log1p )
    INSTANTIATE(log10 )

    INSTANTIATE(sqrt  )
    INSTANTIATE(cbrt  )

    INSTANTIATE(iszero)
    INSTANTIATE(isinf )
    INSTANTIATE(isnan )

    INSTANTIATE(tgamma)
    INSTANTIATE(lgamma)
}
