/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af {
array anisotropicDiffusion(const array& in, const float timestep,
                           const float conductance, const unsigned iterations,
                           const fluxFunction fftype, const diffusionEq eq) {
    af_array out = 0;
    AF_THROW(af_anisotropic_diffusion(&out, in.get(), timestep, conductance,
                                      iterations, fftype, eq));
    return array(out);
}
}  // namespace af
