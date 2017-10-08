/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/ArrayInfo.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <common/err_common.hpp>

#include <backend.hpp>
#include <platform.hpp>

using af::dim4;

dim4 calcStrides(const dim4 &parentDim)
{
    dim4 out(1, 1, 1, 1);
    dim_t *out_dims = out.get();
    const dim_t *parent_dims =  parentDim.get();

    for (dim_t i=1; i < 4; i++) {
        out_dims[i] = out_dims[i - 1] * parent_dims[i-1];
    }

    return out;
}

int ArrayInfo::getDevId() const
{
    // The actual device ID is only stored in the first 8 bits of devId
    // See ArrayInfo.hpp for more
    return devId & 0xff;
}

void ArrayInfo::setId(int id) const
{
    // 1 << (backendId + 8) sets the 9th, 10th or 11th bit of devId to 1
    // for CPU, CUDA and OpenCL respectively
    // See ArrayInfo.hpp for more
    int backendId = detail::getBackend() >> 1; // Convert enums 1, 2, 4 to ints 0, 1, 2
    const_cast<ArrayInfo *>(this)->setId(id | 1 << (backendId + 8));
}

void ArrayInfo::setId(int id)
{
    // 1 << (backendId + 8) sets the 9th, 10th or 11th bit of devId to 1
    // for CPU, CUDA and OpenCL respectively
    // See ArrayInfo.hpp for more
    int backendId = detail::getBackend() >> 1; // Convert enums 1, 2, 4 to ints 0, 1, 2
    devId = id | 1 << (backendId + 8);
}

af_backend ArrayInfo::getBackendId() const
{
    // devId >> 8 converts the backend info to 1, 2, 4 which are enums
    // for CPU, CUDA and OpenCL respectively
    // See ArrayInfo.hpp for more
    int backendId = devId >> 8;
    return (af_backend)backendId;
}

void ArrayInfo::modStrides(const dim4 &newStrides)
{
    dim_strides = newStrides;
}

void ArrayInfo::modDims(const dim4 &newDims)
{
    dim_size   = newDims;
    modStrides(calcStrides(newDims));
}

bool ArrayInfo::isEmpty() const
{
    return (elements() == 0);
}

bool ArrayInfo::isScalar() const
{
    return (elements() == 1);
}

bool ArrayInfo::isRow() const
{
    return (dims()[0] == 1 && dims()[1] > 1 && dims()[2] == 1 && dims()[3] == 1);
}

bool ArrayInfo::isColumn() const
{
    return (dims()[0] > 1 && dims()[1] == 1 && dims()[2] == 1 && dims()[3] == 1);
}

bool ArrayInfo::isVector() const
{
    int singular_dims = 0;
    int non_singular_dims = 0;
    for(int i = 0; i < AF_MAX_DIMS; i++) {
        non_singular_dims += (dims()[i] != 0 && dims()[i] != 1);
        singular_dims += (dims()[i] == 1);
    }
    return singular_dims == AF_MAX_DIMS - 1 && non_singular_dims == 1;
}

bool ArrayInfo::isComplex() const
{
    return ((type == c32) || (type == c64));
}

bool ArrayInfo::isReal() const
{
    return !isComplex();
}

bool ArrayInfo::isDouble() const
{
    return (type == f64 || type == c64);
}

bool ArrayInfo::isSingle() const
{
    return (type == f32 || type == c32);
}

bool ArrayInfo::isRealFloating() const
{
    return (type == f64 || type == f32);
}

bool ArrayInfo::isFloating() const
{
    return (!isInteger() && !isBool());
}

bool ArrayInfo::isInteger() const
{
    return (type == s32
         || type == u32
         || type == s64
         || type == u64
         || type == s16
         || type == u16
         || type == u8);
}

bool ArrayInfo::isBool() const
{
    return (type == b8);
}

bool ArrayInfo::isLinear() const
{
    if (ndims() == 1) {
        return dim_strides[0] == 1;
    }

    dim_t count = 1;
    for (int i = 0; i < (int)ndims(); i++) {
        if (count != dim_strides[i]) {
            return false;
        }
        count *= dim_size[i];
    }
    return true;
}

bool ArrayInfo::isSparse() const
{
    return is_sparse;
}

dim4 getOutDims(const dim4 &ldims, const dim4 &rdims, bool batchMode)
{
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

dim4
toDims(const vector<af_seq>& seqs, const dim4 &parentDims)
{
    dim4 outDims(1, 1, 1, 1);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        outDims[i] = af::calcDim(seqs[i], parentDims[i]);
        if (outDims[i] > parentDims[i])
            AF_ERROR("Size mismatch between input and output", AF_ERR_SIZE);
    }
    return outDims;
}

dim4
toOffset(const vector<af_seq>& seqs, const dim4 &parentDims)
{
    dim4 outOffsets(0, 0, 0, 0);
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if (seqs[i].step !=0 && seqs[i].begin >= 0) {
            outOffsets[i] = seqs[i].begin;
        } else if (seqs[i].begin <= -1) {
            outOffsets[i] = parentDims[i] + seqs[i].begin;
        } else {
            outOffsets[i] = 0;
        }

        if (outOffsets[i] >= parentDims[i])
            AF_ERROR("Index out of range", AF_ERR_SIZE);
    }
    return outOffsets;
}

dim4
toStride(const vector<af_seq>& seqs, const af::dim4 &parentDims)
{
    dim4 out(calcStrides(parentDims));
    for(unsigned i = 0; i < seqs.size(); i++ ) {
        if  (seqs[i].step != 0) {   out[i] *= seqs[i].step; }
    }
    return out;
}

const ArrayInfo&
getInfo(const af_array arr, bool sparse_check, bool device_check)
{
  const ArrayInfo *info = static_cast<ArrayInfo*>(reinterpret_cast<void *>(arr));

  // Check Sparse -> If false, then both standard Array<T> and SparseArray<T> are accepted
  // Otherwise only regular Array<T> is accepted
  if(sparse_check) {
    ARG_ASSERT(0, info->isSparse() == false);
  }

  if (device_check && info->getDevId() != detail::getActiveDeviceId()) {
    AF_ERROR("Input Array not created on current device", AF_ERR_DEVICE);
  }

  return *info;
}
