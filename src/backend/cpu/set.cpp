/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <algorithm>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <set.hpp>
#include <copy.hpp>
#include <sort.hpp>
#include <err_cpu.hpp>
#include <vector>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
{

using namespace std;
using af::dim4;

template<typename T>
Array<T> setUnique(const Array<T> &in,
                    const bool is_sorted)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(af::dim4());
    if (is_sorted) out = copyArray<T>(in);
    else           out = sort<T>(in, 0, true);

    // Need to sync old jobs since we need to
    // operator on pointers directly in std::unique
    getQueue().sync();

    T *ptr = out.get();
    T *last = std::unique(ptr, ptr + in.elements());
    dim_t dist = (dim_t)std::distance(ptr, last);

    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);
    return out;
}

template<typename T>
Array<T> setUnion(const Array<T> &first,
                   const Array<T> &second,
                   const bool is_unique)
{
    first.eval();
    second.eval();
    getQueue().sync();

    Array<T> uFirst = first;
    Array<T> uSecond = second;

    if (!is_unique) {
        // FIXME: Perhaps copy + unique would do ?
        uFirst  = setUnique(first, false);
        uSecond = setUnique(second, false);
    }

    dim_t first_elements  = uFirst.elements();
    dim_t second_elements = uSecond.elements();
    dim_t elements = first_elements + second_elements;

    Array<T> out = createEmptyArray<T>(af::dim4(elements));

    T *ptr = out.get();
    T *last = std::set_union(uFirst.get() , uFirst.get()  + first_elements,
                             uSecond.get(), uSecond.get() + second_elements,
                             ptr);

    dim_t dist = (dim_t)std::distance(ptr, last);
    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);

    return out;
}

template<typename T>
Array<T> setIntersect(const Array<T> &first,
                      const Array<T> &second,
                      const bool is_unique)
{
    first.eval();
    second.eval();
    getQueue().sync();

    Array<T> uFirst = first;
    Array<T> uSecond = second;

    if (!is_unique) {
        uFirst  = setUnique(first, false);
        uSecond = setUnique(second, false);
    }

    dim_t first_elements  = uFirst.elements();
    dim_t second_elements = uSecond.elements();
    dim_t elements = std::max(first_elements, second_elements);

    Array<T> out = createEmptyArray<T>(af::dim4(elements));

    T *ptr = out.get();
    T *last = std::set_intersection(uFirst.get() , uFirst.get()  + first_elements,
                                    uSecond.get(), uSecond.get() + second_elements,
                                    ptr);

    dim_t dist = (dim_t)std::distance(ptr, last);
    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);

    return out;
}

#define INSTANTIATE(T)                                                  \
    template Array<T> setUnique<T>(const Array<T> &in, const bool is_sorted); \
    template Array<T> setUnion<T>(const Array<T> &first, const Array<T> &second, const bool is_unique); \
    template Array<T> setIntersect<T>(const Array<T> &first, const Array<T> &second, const bool is_unique); \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}
