/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <anisotropic_diffusion.hpp>
#include <kernel/anisotropic_diffusion.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace cuda {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const af::fluxFunction fftype,
                          const af::diffusionEq eq) {
    kernel::anisotropicDiffusion<T>(inout, dt, mct, fftype,
                                    eq == AF_DIFFUSION_MCDE);
}

#define INSTANTIATE(T)                                     \
    template void anisotropicDiffusion<T>(                 \
        Array<T> & inout, const float dt, const float mct, \
        const af::fluxFunction fftype, const af::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE(float)
}  // namespace cuda
}  // namespace arrayfire
