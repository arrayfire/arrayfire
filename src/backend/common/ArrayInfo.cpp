/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/traits.hpp>
#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

#include <backend.hpp>
#include <platform.hpp>

using af::dim4;

dim4 calcStrides(const dim4 &parentDim) {
    dim4 out(1, 1, 1, 1);
    dim_t *out_dims          = out.get();
    const dim_t *parent_dims = parentDim.get();

    for (dim_t i = 1; i < 4; i++) {
        out_dims[i] = out_dims[i - 1] * parent_dims[i - 1];
    }

    return out;
}

ArrayInfo::ArrayInfo(unsigned id, af::dim4 size, dim_t offset_, af::dim4 stride,
                     af_dtype af_type)
    : devId(id)
    , type(af_type)
    , dim_size(size)
    , offset(offset_)
    , dim_strides(stride)
    , is_sparse(false) {
    setId(id);
    static_assert(std::is_move_assignable<ArrayInfo>::value,
                  "ArrayInfo is not move assignable");
    static_assert(std::is_move_constructible<ArrayInfo>::value,
                  "ArrayInfo is not move constructible");
    static_assert(
        offsetof(ArrayInfo, devId) == 0,
        "ArrayInfo::devId must be the first member variable of ArrayInfo. \
                   devId is used to encode the backend into the integer. \
                   This is then used in the unified backend to check mismatched arrays.");
    static_assert(std::is_standard_layout<ArrayInfo>::value,
                  "ArrayInfo must be a standard layout type");
}

ArrayInfo::ArrayInfo(unsigned id, af::dim4 size, dim_t offset_, af::dim4 stride,
                     af_dtype af_type, bool sparse)
    : devId(id)
    , type(af_type)
    , dim_size(size)
    , offset(offset_)
    , dim_strides(stride)
    , is_sparse(sparse) {
    setId(id);
    static_assert(
        offsetof(ArrayInfo, devId) == 0,
        "ArrayInfo::devId must be the first member variable of ArrayInfo. \
                   devId is used to encode the backend into the integer. \
                   This is then used in the unified backend to check mismatched arrays.");
    static_assert(std::is_nothrow_move_assignable<ArrayInfo>::value,
                  "ArrayInfo is not nothrow move assignable");
    static_assert(std::is_nothrow_move_constructible<ArrayInfo>::value,
                  "ArrayInfo is not nothrow move constructible");
}

unsigned ArrayInfo::getDevId() const {
    // The actual device ID is only stored in the first 8 bits of devId
    // See ArrayInfo.hpp for more
    return devId & 0xffU;
}

void ArrayInfo::setId(int id) const {
    const_cast<ArrayInfo *>(this)->setId(id);
}

void ArrayInfo::setId(int id) {
    /// Shift the backend flag to the end of the devId integer
    unsigned backendId = detail::getBackend();
    devId              = id | backendId << 8U;
}

af_backend ArrayInfo::getBackendId() const {
    // devId >> 8 converts the backend info to 1, 2, 4 which are enums
    // for CPU, CUDA, OpenCL, and oneAPI respectively
    // See ArrayInfo.hpp for more
    unsigned backendId = devId >> 8U;
    return static_cast<af_backend>(backendId);
}

void ArrayInfo::modStrides(const dim4 &newStrides) { dim_strides = newStrides; }

void ArrayInfo::modDims(const dim4 &newDims) {
    dim_size = newDims;
    modStrides(calcStrides(newDims));
}

bool ArrayInfo::isEmpty() const { return (elements() == 0); }

bool ArrayInfo::isScalar() const { return (elements() == 1); }

bool ArrayInfo::isRow() const {
    return (dims()[0] == 1 && dims()[1] > 1 && dims()[2] == 1 &&
            dims()[3] == 1);
}

bool ArrayInfo::isColumn() const {
    return (dims()[0] > 1 && dims()[1] == 1 && dims()[2] == 1 &&
            dims()[3] == 1);
}

bool ArrayInfo::isVector() const {
    int singular_dims     = 0;
    int non_singular_dims = 0;
    for (int i = 0; i < AF_MAX_DIMS; i++) {
        non_singular_dims += (dims()[i] != 0 && dims()[i] != 1);
        singular_dims += (dims()[i] == 1);
    }
    return singular_dims == AF_MAX_DIMS - 1 && non_singular_dims == 1;
}

bool ArrayInfo::isComplex() const { return arrayfire::common::isComplex(type); }

bool ArrayInfo::isReal() const { return arrayfire::common::isReal(type); }

bool ArrayInfo::isDouble() const { return arrayfire::common::isDouble(type); }

bool ArrayInfo::isSingle() const { return arrayfire::common::isSingle(type); }

bool ArrayInfo::isHalf() const { return arrayfire::common::isHalf(type); }

bool ArrayInfo::isRealFloating() const {
    return arrayfire::common::isRealFloating(type);
}

bool ArrayInfo::isFloating() const {
    return arrayfire::common::isFloating(type);
}

bool ArrayInfo::isInteger() const { return arrayfire::common::isInteger(type); }

bool ArrayInfo::isBool() const { return arrayfire::common::isBool(type); }

bool ArrayInfo::isLinear() const {
    if (ndims() == 1) { return dim_strides[0] == 1; }

    dim_t count = 1;
    for (dim_t i = 0; i < ndims(); i++) {
        if (count != dim_strides[i]) { return false; }
        count *= dim_size[i];
    }
    return true;
}

bool ArrayInfo::isSparse() const { return is_sparse; }

dim4 getOutDims(const dim4 &ldims, const dim4 &rdims, bool batchMode) {
    if (!batchMode) {
        DIM_ASSERT(1, ldims == rdims);
        return ldims;
    }

    dim_t odims[] = {1, 1, 1, 1};
    for (int i = 0; i < 4; i++) {
        DIM_ASSERT(1, ldims[i] == rdims[i] || ldims[i] == 1 || rdims[i] == 1);
        odims[i] = std::max(ldims[i], rdims[i]);
    }

    return dim4(4, odims);
}

using std::vector;

dim4 toDims(const vector<af_seq> &seqs, const dim4 &parentDims) {
    dim4 outDims(1, 1, 1, 1);
    for (unsigned i = 0; i < seqs.size(); i++) {
        outDims[i] = af::calcDim(seqs[i], parentDims[i]);
        if (outDims[i] > parentDims[i]) {
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
        }
    }
    return outDims;
}

dim4 toOffset(const vector<af_seq> &seqs, const dim4 &parentDims) {
    dim4 outOffsets(0, 0, 0, 0);
    for (unsigned i = 0; i < seqs.size(); i++) {
        if (seqs[i].step != 0 && seqs[i].begin >= 0) {
            outOffsets[i] = seqs[i].begin;
        } else if (seqs[i].begin <= -1) {
            outOffsets[i] = parentDims[i] + seqs[i].begin;
        } else {
            outOffsets[i] = 0;
        }

        if (outOffsets[i] >= parentDims[i]) {
            AF_ERROR("Index out of range", AF_ERR_SIZE);
        }
    }
    return outOffsets;
}

dim4 toStride(const vector<af_seq> &seqs, const af::dim4 &parentDims) {
    dim4 out(calcStrides(parentDims));
    for (unsigned i = 0; i < seqs.size(); i++) {
        if (seqs[i].step != 0) { out[i] *= seqs[i].step; }
    }
    return out;
}

namespace arrayfire {
namespace common {

const ArrayInfo &getInfo(const af_array arr, bool sparse_check) {
    const ArrayInfo *info = nullptr;
    memcpy(&info, &arr, sizeof(af_array));

    // Check Sparse -> If false, then both standard Array<T> and SparseArray<T>
    // are accepted Otherwise only regular Array<T> is accepted
    if (sparse_check) { ARG_ASSERT(0, info->isSparse() == false); }

    return *info;
}

}  // namespace common
}  // namespace arrayfire
