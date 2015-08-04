/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <handle.hpp>
#include <fft.hpp>
#include <copy.hpp>

using namespace detail;

static void computePaddedDims(dim4 &pdims,
                              const dim4 &idims,
                              const dim_t npad,
                              dim_t const * const pad)
{
    for (int i = 0; i < 4; i++) {
        pdims[i] = (i < (int)npad) ? pad[i] : idims[i];
    }
}

template<typename inType, typename outType, int rank, bool direction>
Array<outType> fft(const Array<inType> input, const double norm_factor,
                   const dim_t npad, const dim_t  * const pad)
{
    dim4 pdims(1);
    computePaddedDims(pdims, input.dims(), npad, pad);
    Array<outType> output = padArray<inType, outType>(input, pdims, scalar<outType>(0), norm_factor);

    fft_inplace<outType, rank, direction>(output);

    return output;
}
