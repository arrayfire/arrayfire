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

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> fft(Array<inType> const &in, double norm_factor, dim_t const npad, dim_t const * const pad);

template<typename T, int rank>
Array<T> ifft(Array<T> const &in, double norm_factor, dim_t const npad, dim_t const * const pad);

template<typename T, int rank, bool direction>
void fft_common(Array<T> &out, const Array<T> &in);

}
