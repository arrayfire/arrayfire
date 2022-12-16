/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <debug_cuda.hpp>
#include <kernel/thrust_sort_by_key.hpp>
#include <thrust/sort.h>
#include <thrust_utils.hpp>
#include <types.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {
// Wrapper functions
template<typename Tk, typename Tv>
void thrustSortByKey(Tk *keyPtr, Tv *valPtr, int elements, bool isAscending) {
    if (isAscending) {
        THRUST_SELECT(thrust::stable_sort_by_key, keyPtr, keyPtr + elements,
                      valPtr);
    } else {
        THRUST_SELECT(thrust::stable_sort_by_key, keyPtr, keyPtr + elements,
                      valPtr, thrust::greater<Tk>());
    }
    POST_LAUNCH_CHECK();
}

#define INSTANTIATE(Tk, Tv)                                         \
    template void thrustSortByKey<Tk, Tv>(Tk * keyPtr, Tv * valPtr, \
                                          int elements, bool isAscending);

#define INSTANTIATE0(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, uchar)

#define INSTANTIATE1(Tk)    \
    INSTANTIATE(Tk, int)    \
    INSTANTIATE(Tk, uint)   \
    INSTANTIATE(Tk, short)  \
    INSTANTIATE(Tk, ushort) \
    INSTANTIATE(Tk, intl)   \
    INSTANTIATE(Tk, uintl)

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
