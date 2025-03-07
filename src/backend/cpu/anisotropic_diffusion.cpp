/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <anisotropic_diffusion.hpp>
#include <kernel/anisotropic_diffusion.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cpu {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const af::fluxFunction fftype,
                          const af::diffusionEq eq) {
    if (eq == AF_DIFFUSION_MCDE) {
        getQueue().enqueue(kernel::anisotropicDiffusion<T, true>, inout, dt,
                           mct, fftype);
    } else {
        getQueue().enqueue(kernel::anisotropicDiffusion<T, false>, inout, dt,
                           mct, fftype);
    }
}

#define INSTANTIATE(T)                                     \
    template void anisotropicDiffusion<T>(                 \
        Array<T> & inout, const float dt, const float mct, \
        const af::fluxFunction fftype, const af::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE(float)
}  // namespace cpu
}  // namespace arrayfire
