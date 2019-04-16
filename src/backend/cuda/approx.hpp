/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cuda {
template<typename Ty, typename Tp>
Array<Ty> approx1(const Array<Ty> &in, const Array<Tp> &pos,
                  const af_interp_type method, const float offGrid);

template<typename Ty, typename Tp>
Array<Ty> approx2(const Array<Ty> &in, const Array<Tp> &pos0,
                  const Array<Tp> &pos1, const af_interp_type method,
                  const float offGrid);
}  // namespace cuda
