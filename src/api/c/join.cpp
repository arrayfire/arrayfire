/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <af/data.h>

#include <algorithm>
#include <climits>
#include <vector>

using af::dim4;
using arrayfire::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::swap;
using std::vector;

template<typename T>
static inline af_array join(const int dim, const af_array first,
                            const af_array second) {
    return getHandle(join<T>(dim, getArray<T>(first), getArray<T>(second)));
}

template<typename T>
static inline af_array join_many(const int dim, const unsigned n_arrays,
                                 const af_array *inputs) {
    vector<Array<T>> inputs_;
    inputs_.reserve(n_arrays);

    dim_t dim_size{0};
    for (unsigned i{0}; i < n_arrays; ++i) {
        const Array<T> &iArray = getArray<T>(inputs[i]);
        if (!iArray.isEmpty()) {
            inputs_.push_back(iArray);
            dim_size += iArray.dims().dims[dim];
        }
    }

    // All dimensions except join dimension must be equal
    // calculate odims size
    af::dim4 odims{inputs_[0].dims()};
    odims.dims[dim] = dim_size;

    Array<T> out{createEmptyArray<T>(odims)};
    join<T>(out, dim, inputs_);
    return getHandle(out);
}

af_err af_join(af_array *out, const int dim, const af_array first,
               const af_array second) {
    try {
        const ArrayInfo &finfo{getInfo(first)};
        const ArrayInfo &sinfo{getInfo(second)};
        const dim4 &fdims{finfo.dims()};
        const dim4 &sdims{sinfo.dims()};

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(2, finfo.getType() == sinfo.getType());
        if (sinfo.elements() == 0) { return af_retain_array(out, first); }
        if (finfo.elements() == 0) { return af_retain_array(out, second); }
        DIM_ASSERT(2, finfo.elements() > 0);
        DIM_ASSERT(3, sinfo.elements() > 0);

        // All dimensions except join dimension must be equal
        for (int i{0}; i < AF_MAX_DIMS; i++) {
            if (i != dim) { DIM_ASSERT(2, fdims.dims[i] == sdims.dims[i]); }
        }

        af_array output;

        switch (finfo.getType()) {
            case f32: output = join<float>(dim, first, second); break;
            case c32: output = join<cfloat>(dim, first, second); break;
            case f64: output = join<double>(dim, first, second); break;
            case c64: output = join<cdouble>(dim, first, second); break;
            case b8: output = join<char>(dim, first, second); break;
            case s32: output = join<int>(dim, first, second); break;
            case u32: output = join<uint>(dim, first, second); break;
            case s64: output = join<intl>(dim, first, second); break;
            case u64: output = join<uintl>(dim, first, second); break;
            case s16: output = join<short>(dim, first, second); break;
            case u16: output = join<ushort>(dim, first, second); break;
            case u8: output = join<uchar>(dim, first, second); break;
            case f16: output = join<half>(dim, first, second); break;
            default: TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays,
                    const af_array *inputs) {
    try {
        ARG_ASSERT(3, inputs != nullptr);

        if (n_arrays == 1) {
            af_array ret{nullptr};
            AF_CHECK(af_retain_array(&ret, *inputs));
            std::swap(*out, ret);
            return AF_SUCCESS;
        }

        ARG_ASSERT(1, dim >= 0 && dim < AF_MAX_DIMS);
        ARG_ASSERT(2, n_arrays > 0);

        const af_array *inputIt{inputs};
        const af_array *inputEnd{inputs + n_arrays};
        while ((inputIt != inputEnd) && (getInfo(*inputIt).elements() == 0)) {
            ++inputIt;
        }
        if (inputIt == inputEnd) {
            // All arrays have 0 elements
            af_array ret = nullptr;
            AF_CHECK(af_retain_array(&ret, *inputs));
            std::swap(*out, ret);
            return AF_SUCCESS;
        }

        // inputIt points to first non empty array
        const af_dtype assertType{getInfo(*inputIt).getType()};
        const dim4 &assertDims{getInfo(*inputIt).dims()};

        // Check all remaining arrays on assertType and assertDims
        while (++inputIt != inputEnd) {
            const ArrayInfo &info = getInfo(*inputIt);
            if (info.elements() > 0) {
                ARG_ASSERT(3, assertType == info.getType());
                const dim4 &infoDims{getInfo(*inputIt).dims()};
                // All dimensions except join dimension must be equal
                for (int i{0}; i < AF_MAX_DIMS; i++) {
                    if (i != dim) {
                        DIM_ASSERT(3, assertDims.dims[i] == infoDims.dims[i]);
                    }
                }
            }
        }
        af_array output;

        switch (assertType) {
            case f32: output = join_many<float>(dim, n_arrays, inputs); break;
            case c32: output = join_many<cfloat>(dim, n_arrays, inputs); break;
            case f64: output = join_many<double>(dim, n_arrays, inputs); break;
            case c64: output = join_many<cdouble>(dim, n_arrays, inputs); break;
            case b8: output = join_many<char>(dim, n_arrays, inputs); break;
            case s32: output = join_many<int>(dim, n_arrays, inputs); break;
            case u32: output = join_many<uint>(dim, n_arrays, inputs); break;
            case s64: output = join_many<intl>(dim, n_arrays, inputs); break;
            case u64: output = join_many<uintl>(dim, n_arrays, inputs); break;
            case s16: output = join_many<short>(dim, n_arrays, inputs); break;
            case u16: output = join_many<ushort>(dim, n_arrays, inputs); break;
            case u8: output = join_many<uchar>(dim, n_arrays, inputs); break;
            case f16: output = join_many<half>(dim, n_arrays, inputs); break;
            default: TYPE_ERROR(1, assertType);
        }
        swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
