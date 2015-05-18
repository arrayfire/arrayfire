/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <set.hpp>
#include <copy.hpp>
#include <sort.hpp>
#include <err_cuda.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>

namespace cuda
{
    using af::dim4;

    template<typename T>
    Array<T> setUnique(const Array<T> &in,
                        const bool is_sorted)
    {
        Array<T> out = copyArray<T>(in);

        thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());
        thrust::device_ptr<T> out_ptr_end = out_ptr + out.dims()[0];

        if(!is_sorted) thrust::sort(out_ptr, out_ptr_end);
        thrust::device_ptr<T> out_ptr_last = thrust::unique(out_ptr, out_ptr_end);

        out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));
        return out;
    }

    template<typename T>
    Array<T> setUnion(const Array<T> &first,
                       const Array<T> &second,
                       const bool is_unique)
    {
        Array<T> unique_first = first;
        Array<T> unique_second = second;

        if (!is_unique) {
            unique_first = setUnique(first, false);
            unique_second = setUnique(second, false);
        }

        dim_t out_size = unique_first.dims()[0] + unique_second.dims()[0];
        Array<T> out = createEmptyArray<T>(dim4(out_size));

        thrust::device_ptr<T> first_ptr = thrust::device_pointer_cast<T>(unique_first.get());
        thrust::device_ptr<T> first_ptr_end = first_ptr + unique_first.dims()[0];

        thrust::device_ptr<T> second_ptr = thrust::device_pointer_cast<T>(unique_second.get());
        thrust::device_ptr<T> second_ptr_end = second_ptr + unique_second.dims()[0];

        thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());

        thrust::device_ptr<T> out_ptr_last = thrust::set_union(first_ptr, first_ptr_end,
                                                               second_ptr, second_ptr_end,
                                                               out_ptr);

        out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));

        return out;
    }

    template<typename T>
    Array<T> setIntersect(const Array<T> &first,
                           const Array<T> &second,
                           const bool is_unique)
    {
        Array<T> unique_first = first;
        Array<T> unique_second = second;

        if (!is_unique) {
            unique_first = setUnique(first, false);
            unique_second = setUnique(second, false);
        }

        dim_t out_size = std::max(unique_first.dims()[0], unique_second.dims()[0]);
        Array<T> out = createEmptyArray<T>(dim4(out_size));

        thrust::device_ptr<T> first_ptr = thrust::device_pointer_cast<T>(unique_first.get());
        thrust::device_ptr<T> first_ptr_end = first_ptr + unique_first.dims()[0];

        thrust::device_ptr<T> second_ptr = thrust::device_pointer_cast<T>(unique_second.get());
        thrust::device_ptr<T> second_ptr_end = second_ptr + unique_second.dims()[0];

        thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());

        thrust::device_ptr<T> out_ptr_last = thrust::set_intersection(first_ptr, first_ptr_end,
                                                                      second_ptr, second_ptr_end,
                                                                      out_ptr);

        out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));

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
}
