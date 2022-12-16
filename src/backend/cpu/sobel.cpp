/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve.hpp>
#include <kernel/sobel.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sobel.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename Ti, typename To>
std::pair<Array<To>, Array<To>> sobelDerivatives(const Array<Ti> &img,
                                                 const unsigned &ker_size) {
    UNUSED(ker_size);
    // ket_size is for future proofing, this argument is not used
    // currently
    Array<To> dx = createEmptyArray<To>(img.dims());
    Array<To> dy = createEmptyArray<To>(img.dims());

    getQueue().enqueue(kernel::derivative<Ti, To, true>, dx, img);
    getQueue().enqueue(kernel::derivative<Ti, To, false>, dy, img);

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(Ti, To)                                    \
    template std::pair<Array<To>, Array<To>> sobelDerivatives( \
        const Array<Ti> &img, const unsigned &ker_size);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(int, int)
INSTANTIATE(uint, int)
INSTANTIATE(char, int)
INSTANTIATE(uchar, int)
INSTANTIATE(short, int)
INSTANTIATE(ushort, int)

}  // namespace cpu
}  // namespace arrayfire
