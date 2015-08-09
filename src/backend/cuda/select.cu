/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <select.hpp>
#include <err_cuda.hpp>
#include <kernel/select.hpp>

namespace cuda
{
    template<typename T>
    void select(Array<T> &out, const Array<char> &cond, const Array<T> &a, const Array<T> &b)
    {
        kernel::select<T>(out, cond, a, b, out.ndims());
    }

    template<typename T, bool flip>
    void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a, const double &b)
    {
        kernel::select_scalar<T, flip>(out, cond, a, b, out.ndims());
    }

#define INSTANTIATE(T)                                              \
    template void select<T>(Array<T> &out, const Array<char> &cond, \
                            const Array<T> &a, const Array<T> &b);  \
    template void select_scalar<T, true >(Array<T> &out,            \
                                          const Array<char> &cond,  \
                                          const Array<T> &a,        \
                                          const double &b);         \
    template void select_scalar<T, false>(Array<T> &out, const      \
                                          Array<char> &cond,        \
                                          const Array<T> &a,        \
                                          const double &b);         \

    INSTANTIATE(float  )
    INSTANTIATE(double )
    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )
    INSTANTIATE(char   )
    INSTANTIATE(uchar  )
}
