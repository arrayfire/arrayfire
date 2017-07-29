/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <anisotropic_diffusion.hpp>
#include <kernel/anisotropic_diffusion.hpp>

namespace opencl
{
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt,
                          const float mct, const af::fluxFunction fftype,
                          const af::diffusionEq eq)
{
    if (eq==AF_DIFFUSION_MCDE)
        kernel::anisotropicDiffusion<T, true>(inout, dt, mct, fftype);
    else
        kernel::anisotropicDiffusion<T, false>(inout, dt, mct, fftype);
}

#define INSTANTIATE(T)\
template void anisotropicDiffusion<T>(Array<T> &inout, const float dt, const float mct,\
                                      const af::fluxFunction fftype, const af::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE( float)
}
