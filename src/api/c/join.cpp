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
#include <vector>

using af::dim4;
using common::half;
using namespace detail;

template<typename T>
static inline af_array join(const int dim, const af_array first,
                            const af_array second) {
    return getHandle(join<T>(dim, getArray<T>(first), getArray<T>(second)));
}

template<typename T>
static inline af_array join_many(const int dim, const unsigned n_arrays,
                                 const af_array *inputs) {
    std::vector<Array<T>> inputs_;
    inputs_.reserve(n_arrays);

    for (unsigned i = 0; i < n_arrays; i++) {
        inputs_.push_back(getArray<T>(inputs[i]));
    }
    return getHandle(join<T>(dim, inputs_));
}

af_err af_join(af_array *out, const int dim, const af_array first,
               const af_array second) {
    try {
        const ArrayInfo &finfo = getInfo(first);
        const ArrayInfo &sinfo = getInfo(second);
        af::dim4 fdims         = finfo.dims();
        af::dim4 sdims         = sinfo.dims();

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(2, finfo.getType() == sinfo.getType());
        if (sinfo.elements() == 0) { return af_retain_array(out, first); }

        if (finfo.elements() == 0) { return af_retain_array(out, second); }

        DIM_ASSERT(2, sinfo.elements() > 0);
        DIM_ASSERT(3, finfo.elements() > 0);

        // All dimensions except join dimension must be equal
        // Compute output dims
        for (int i = 0; i < 4; i++) {
            if (i != dim) { DIM_ASSERT(2, fdims[i] == sdims[i]); }
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
            af_array ret = nullptr;
            AF_CHECK(af_retain_array(&ret, inputs[0]));
            std::swap(*out, ret);
            return AF_SUCCESS;
        }

        std::vector<ArrayInfo> info;
        info.reserve(n_arrays);
        std::vector<af::dim4> dims(n_arrays);
        for (unsigned i = 0; i < n_arrays; i++) {
            info.push_back(getInfo(inputs[i]));
            dims[i] = info[i].dims();
        }

        ARG_ASSERT(1, dim >= 0 && dim < 4);

        for (unsigned i = 1; i < n_arrays; i++) {
            ARG_ASSERT(3, info[0].getType() == info[i].getType());
            DIM_ASSERT(3, info[i].elements() > 0);
        }

        // All dimensions except join dimension must be equal
        // Compute output dims
        for (int i = 0; i < 4; i++) {
            if (i != dim) {
                for (unsigned j = 1; j < n_arrays; j++) {
                    DIM_ASSERT(3, dims[0][i] == dims[j][i]);
                }
            }
        }

        af_array output;

        switch (info[0].getType()) {
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
            default: TYPE_ERROR(1, info[0].getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
