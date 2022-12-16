/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flood_fill.hpp>

#include <err_opencl.hpp>
#include <kernel/flood_fill.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const af::connectivity nlookup) {
    auto out = createValueArray(image.dims(), T(0));
    kernel::floodFill<T>(out, image, seedsX, seedsY, newValue, lowValue,
                         highValue, nlookup);
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> floodFill(const Array<T>&, const Array<uint>&,           \
                                const Array<uint>&, const T, const T, const T, \
                                const af::connectivity);

INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace opencl
}  // namespace arrayfire
