/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl
{
    template<typename Ty, typename Tp>
    Array<Ty> approx1(const Array<Ty> &yi,
                      const Array<Tp> &xo, const int xdim,
                      const Tp &xi_beg, const Tp &xi_step,
                      const af_interp_type method, const float offGrid);

    template<typename Ty, typename Tp>
    Array<Ty> approx2(const Array<Ty> &zi,
                      const Array<Tp> &xo, const int xdim,
                      const Array<Tp> &yo, const int ydim,
                      const Tp &xi_beg, const Tp &xi_step,
                      const Tp &yi_beg, const Tp &yi_step,
                      const af_interp_type method, const float offGrid);
}
