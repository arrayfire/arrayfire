/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <ireduce.hpp>

#undef _GLIBCXX_USE_INT128
#include <complex>
#include <kernel/ireduce.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

    template<af_op_t op, typename T>
    void ireduce(Array<T> &out, Array<uint> &loc,
                 const Array<T> &in, const int dim)
    {
        kernel::ireduce<T, op>(out, loc.get(), in, dim);
    }

    template<af_op_t op, typename T>
    T ireduce_all(unsigned *loc, const Array<T> &in)
    {
        return kernel::ireduce_all<T, op>(loc, in);
    }

#define INSTANTIATE(ROp, T)                                             \
    template void ireduce<ROp, T>(Array<T> &out, Array<uint> &loc,      \
                                  const Array<T> &in, const int dim);   \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);  \

    //min
    INSTANTIATE(af_min_t, float  )
    INSTANTIATE(af_min_t, double )
    INSTANTIATE(af_min_t, cfloat )
    INSTANTIATE(af_min_t, cdouble)
    INSTANTIATE(af_min_t, int    )
    INSTANTIATE(af_min_t, uint   )
    INSTANTIATE(af_min_t, intl   )
    INSTANTIATE(af_min_t, uintl  )
    INSTANTIATE(af_min_t, char   )
    INSTANTIATE(af_min_t, uchar  )

    //max
    INSTANTIATE(af_max_t, float  )
    INSTANTIATE(af_max_t, double )
    INSTANTIATE(af_max_t, cfloat )
    INSTANTIATE(af_max_t, cdouble)
    INSTANTIATE(af_max_t, int    )
    INSTANTIATE(af_max_t, uint   )
    INSTANTIATE(af_max_t, intl   )
    INSTANTIATE(af_max_t, uintl  )
    INSTANTIATE(af_max_t, char   )
    INSTANTIATE(af_max_t, uchar  )
}
