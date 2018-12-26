/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/signal.h>
#include "error.hpp"

namespace af {
array approx1(const array &yi, const array &xo, const interpType method,
              const float offGrid) {
    af_array yo = 0;
    AF_THROW(af_approx1(&yo, yi.get(), xo.get(), method, offGrid));
    return array(yo);
}

array approx2(const array &zi, const array &xo, const array &yo,
              const interpType method, const float offGrid) {
    af_array zo = 0;
    AF_THROW(af_approx2(&zo, zi.get(), xo.get(), yo.get(), method, offGrid));
    return array(zo);
}

array approx1(const array &yi, const array &xo, const int xdim,
              const double xi_beg, const double xi_step,
              const interpType method, const float offGrid) {
    af_array yo = 0;
    AF_THROW(af_approx1_uniform(&yo, yi.get(), xo.get(), xdim, xi_beg, xi_step,
                                method, offGrid));
    return array(yo);
}

array approx2(const array &zi, const array &xo, const int xdim,
              const double xi_beg, const double xi_step, const array &yo,
              const int ydim, const double yi_beg, const double yi_step,
              const interpType method, const float offGrid) {
    af_array zo = 0;
    AF_THROW(af_approx2_uniform(&zo, zi.get(), xo.get(), xdim, xi_beg, xi_step,
                                yo.get(), ydim, yi_beg, yi_step, method,
                                offGrid));
    return array(zo);
}
}  // namespace af
