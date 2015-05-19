/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <ArrayInfo.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <err_common.hpp>

using af::dim4;

dim_t
calcOffset(const af::dim4 &strides, const af::dim4 &offsets)
{
    dim_t offset = 0;
    for (int i = 0; i < 4; i++) offset += offsets[i] * strides[i];
    return offset;
}


const ArrayInfo&
getInfo(af_array arr)
{
    const ArrayInfo *info = static_cast<ArrayInfo*>(reinterpret_cast<void *>(arr));
    return *info;
}

af_err
af_get_elements(dim_t *elems, const af_array arr)
{
    *elems =  getInfo(arr).elements();
    return AF_SUCCESS; //FIXME: Catch exceptions correctly
}

af_err af_get_type(af_dtype *type, const af_array arr)
{
    *type = getInfo(arr).getType();
    return AF_SUCCESS; //FIXME: Catch exceptions correctly
}

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
    for(int i = 0; i < AF_MAX_DIMS; i++) {
        singular_dims += (dims()[i] == 1);
    }
    return singular_dims == AF_MAX_DIMS - 1;
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
