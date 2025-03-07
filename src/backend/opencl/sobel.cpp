/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/sobel.hpp>
#include <sobel.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace opencl {

template<typename Ti, typename To>
std::pair<Array<To>, Array<To>> sobelDerivatives(const Array<Ti> &img,
                                                 const unsigned &ker_size) {
    Array<To> dx = createEmptyArray<To>(img.dims());
    Array<To> dy = createEmptyArray<To>(img.dims());

    switch (ker_size) {
        case 3: kernel::sobel<Ti, To, 3>(dx, dy, img); break;
    }

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

}  // namespace opencl
}  // namespace arrayfire
