/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/features.h>
#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/fast_pyramid.hpp>

using af::dim4;
using af::features;

namespace cuda
{

template<typename T>
void fast_pyramid(std::vector<unsigned>& feat_pyr, std::vector<float*>& d_x_pyr,
                  std::vector<float*>& d_y_pyr, std::vector<unsigned>& lvl_best,
                  std::vector<float>& lvl_scl, std::vector<Array<T>>& img_pyr,
                  const Array<T>& image,
                  const float fast_thr, const unsigned max_feat,
                  const float scl_fctr, const unsigned levels,
                  const unsigned patch_size)
{
    kernel::fast_pyramid<T>(feat_pyr, d_x_pyr, d_y_pyr, lvl_best, lvl_scl, img_pyr,
                            image, fast_thr, max_feat, scl_fctr, levels, patch_size);
}

#define INSTANTIATE(T)\
    template void fast_pyramid<T>(std::vector<unsigned>& feat_pyr, std::vector<float*>& d_x_pyr,    \
                                  std::vector<float*>& d_y_pyr, std::vector<unsigned>& lvl_best,    \
                                  std::vector<float>& lvl_scl, std::vector<Array<T>>& img_pyr,      \
                                  const Array<T>& image,                                            \
                                  const float fast_thr, const unsigned max_feat,                    \
                                  const float scl_fctr, const unsigned levels,                      \
                                  const unsigned patch_size);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )
INSTANTIATE(short )
INSTANTIATE(ushort)

}
