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
#include <err_opencl.hpp>
#include <boost/compute/algorithm/set_intersection.hpp>
#include <boost/compute/algorithm/set_union.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/unique.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace compute = boost::compute;

namespace opencl
{
    using af::dim4;

    template<typename T>
    Array<T> setUnique(const Array<T> &in,
                       const bool is_sorted)
    {
        try {
            Array<T> out = copyArray<T>(in);

            compute::command_queue queue(getQueue()());

            compute::buffer out_data((*out.get())());

            compute::buffer_iterator<T> begin(out_data, 0);
            compute::buffer_iterator<T> end(out_data, out.dims()[0]);

            if (!is_sorted) {
                compute::sort(begin, end, queue);
            }

            end = compute::unique(begin, end, queue);

            out.resetDims(dim4(std::distance(begin, end), 1, 1, 1));

            return out;
        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

    template<typename T>
    Array<T> setUnion(const Array<T> &first,
                      const Array<T> &second,
                      const bool is_unique)
    {
        try {
            Array<T> unique_first = first;
            Array<T> unique_second = second;

            if (!is_unique) {
                unique_first  = setUnique(first, false);
                unique_second = setUnique(second, false);
            }

            size_t out_size = unique_first.dims()[0] + unique_second.dims()[0];
            Array<T> out = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

            compute::command_queue queue(getQueue()());

            compute::buffer first_data((*unique_first.get())());
            compute::buffer second_data((*unique_second.get())());
            compute::buffer out_data((*out.get())());

            compute::buffer_iterator<T> first_begin(first_data, 0);
            compute::buffer_iterator<T> first_end(first_data, unique_first.dims()[0]);
            compute::buffer_iterator<T> second_begin(second_data, 0);
            compute::buffer_iterator<T> second_end(second_data, unique_second.dims()[0]);
            compute::buffer_iterator<T> out_begin(out_data, 0);

            compute::buffer_iterator<T> out_end = compute::set_union(
                first_begin, first_end, second_begin, second_end, out_begin, queue
                );

            out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
            return out;

        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
    }

    template<typename T>
    Array<T> setIntersect(const Array<T> &first,
                          const Array<T> &second,
                          const bool is_unique)
    {
        try {
            Array<T> unique_first = first;
            Array<T> unique_second = second;

            if (!is_unique) {
                unique_first  = setUnique(first, false);
                unique_second = setUnique(second, false);
            }

            size_t out_size = std::max(unique_first.dims()[0], unique_second.dims()[0]);
            Array<T> out = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

            compute::command_queue queue(getQueue()());

            compute::buffer first_data((*unique_first.get())());
            compute::buffer second_data((*unique_second.get())());
            compute::buffer out_data((*out.get())());

            compute::buffer_iterator<T> first_begin(first_data, 0);
            compute::buffer_iterator<T> first_end(first_data, unique_first.dims()[0]);
            compute::buffer_iterator<T> second_begin(second_data, 0);
            compute::buffer_iterator<T> second_end(second_data, unique_second.dims()[0]);
            compute::buffer_iterator<T> out_begin(out_data, 0);

            compute::buffer_iterator<T> out_end = compute::set_intersection(
                first_begin, first_end, second_begin, second_end, out_begin, queue
                );

            out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
            return out;
        } catch (std::exception &ex) {
            AF_ERROR(ex.what(), AF_ERR_INTERNAL);
        }
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
