/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <set.hpp>
#include <sort.hpp>
#include <af/dim4.hpp>

AF_DEPRECATED_WARNINGS_OFF
#include <boost/compute/algorithm/set_intersection.hpp>
#include <boost/compute/algorithm/set_union.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/unique.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>
AF_DEPRECATED_WARNINGS_ON

namespace compute = boost::compute;

namespace arrayfire {
namespace opencl {
using af::dim4;

using std::conditional;
using std::is_same;
template<typename T>
using ltype_t = typename conditional<is_same<T, intl>::value, cl_long, T>::type;

template<typename T>
using type_t =
    typename conditional<is_same<T, uintl>::value, cl_ulong, ltype_t<T>>::type;

template<typename T>
Array<T> setUnique(const Array<T> &in, const bool is_sorted) {
    try {
        Array<T> out = copyArray<T>(in);

        compute::command_queue queue(getQueue()());

        compute::buffer out_data((*out.get())());

        compute::buffer_iterator<type_t<T>> begin(out_data, 0);
        compute::buffer_iterator<type_t<T>> end(out_data, out.elements());

        if (!is_sorted) { compute::sort(begin, end, queue); }

        end = compute::unique(begin, end, queue);

        out.resetDims(dim4(std::distance(begin, end), 1, 1, 1));

        return out;
    } catch (const std::exception &ex) { AF_ERROR(ex.what(), AF_ERR_INTERNAL); }
}

template<typename T>
Array<T> setUnion(const Array<T> &first, const Array<T> &second,
                  const bool is_unique) {
    try {
        Array<T> unique_first  = first;
        Array<T> unique_second = second;

        if (!is_unique) {
            unique_first  = setUnique(first, false);
            unique_second = setUnique(second, false);
        }

        size_t out_size = unique_first.elements() + unique_second.elements();
        Array<T> out    = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

        compute::command_queue queue(getQueue()());

        compute::buffer first_data((*unique_first.get())());
        compute::buffer second_data((*unique_second.get())());
        compute::buffer out_data((*out.get())());

        compute::buffer_iterator<type_t<T>> first_begin(first_data, 0);
        compute::buffer_iterator<type_t<T>> first_end(first_data,
                                                      unique_first.elements());
        compute::buffer_iterator<type_t<T>> second_begin(second_data, 0);
        compute::buffer_iterator<type_t<T>> second_end(
            second_data, unique_second.elements());
        compute::buffer_iterator<type_t<T>> out_begin(out_data, 0);

        compute::buffer_iterator<type_t<T>> out_end = compute::set_union(
            first_begin, first_end, second_begin, second_end, out_begin, queue);

        out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
        return out;

    } catch (const std::exception &ex) { AF_ERROR(ex.what(), AF_ERR_INTERNAL); }
}

template<typename T>
Array<T> setIntersect(const Array<T> &first, const Array<T> &second,
                      const bool is_unique) {
    try {
        Array<T> unique_first  = first;
        Array<T> unique_second = second;

        if (!is_unique) {
            unique_first  = setUnique(first, false);
            unique_second = setUnique(second, false);
        }

        size_t out_size =
            std::max(unique_first.elements(), unique_second.elements());
        Array<T> out = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

        compute::command_queue queue(getQueue()());

        compute::buffer first_data((*unique_first.get())());
        compute::buffer second_data((*unique_second.get())());
        compute::buffer out_data((*out.get())());

        compute::buffer_iterator<type_t<T>> first_begin(first_data, 0);
        compute::buffer_iterator<type_t<T>> first_end(first_data,
                                                      unique_first.elements());
        compute::buffer_iterator<type_t<T>> second_begin(second_data, 0);
        compute::buffer_iterator<type_t<T>> second_end(
            second_data, unique_second.elements());
        compute::buffer_iterator<type_t<T>> out_begin(out_data, 0);

        compute::buffer_iterator<type_t<T>> out_end = compute::set_intersection(
            first_begin, first_end, second_begin, second_end, out_begin, queue);

        out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
        return out;
    } catch (const std::exception &ex) { AF_ERROR(ex.what(), AF_ERR_INTERNAL); }
}

#define INSTANTIATE(T)                                                        \
    template Array<T> setUnique<T>(const Array<T> &in, const bool is_sorted); \
    template Array<T> setUnion<T>(                                            \
        const Array<T> &first, const Array<T> &second, const bool is_unique); \
    template Array<T> setIntersect<T>(                                        \
        const Array<T> &first, const Array<T> &second, const bool is_unique);

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
}  // namespace opencl
}  // namespace arrayfire
