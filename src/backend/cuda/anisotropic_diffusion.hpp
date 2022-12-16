/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace cuda {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const af::fluxFunction fftype,
                          const af::diffusionEq eq);
}  // namespace cuda
}  // namespace arrayfire
