/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flood_fill.hpp>

#include <err_oneapi.hpp>

namespace oneapi {

template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const af::connectivity nlookup) {
    ONEAPI_NOT_SUPPORTED("");
    auto out = createValueArray(image.dims(), T(0));
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

}  // namespace oneapi
