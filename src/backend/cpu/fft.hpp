/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

#include <cstddef>

namespace af { class dim4; }

namespace cpu
{

void setFFTPlanCacheSize(size_t numPlans);

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in);

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in);

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims);
}
