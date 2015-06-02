/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <af/util.h>
#include <af/dim4.hpp>
#include <af/device.h>
#include <vector>

dim_t
calcOffset(const af::dim4 &strides, const af::dim4 &offsets);

af::dim4
calcStrides(const af::dim4 &parentDim);

af::dim4 getOutDims(const af::dim4 &ldims, const af::dim4 &rdims, bool batchMode);

/// Array Arrayementation Info class
// This class is the base class to all Array objects. The purpose of this class
// was to have a way to retrieve basic information of an Array object without
// specifying what type the object is at compile time.
class ArrayInfo
{
private:
    int             devId;
    af_dtype        type;
    af::dim4        dim_size;
    af::dim4        dim_offsets, dim_strides;

public:
    ArrayInfo(int id, af::dim4 size, af::dim4 offset, af::dim4 stride, af_dtype af_type):
        devId(id),
        type(af_type),
        dim_size(size),
        dim_offsets(offset),
        dim_strides(stride)
    { af_init(); }

#if __cplusplus > 199711L
    //Copy constructors are deprecated if there is a
    //user-defined destructor in c++11
    ArrayInfo(const ArrayInfo& other) = default;
#endif
    ~ArrayInfo() {}

    const af_dtype& getType() const     { return type;                  }

    const af::dim4& offsets() const     { return dim_offsets;           }

    const af::dim4& strides()    const  { return dim_strides;           }

    size_t elements() const             { return dim_size.elements();   }
    size_t ndims() const                { return dim_size.ndims();      }
    const af::dim4& dims() const        { return dim_size;              }

    int getDevId() const { return devId; }

    void setId(int id) const { const_cast<ArrayInfo *>(this)->setId(id); }
    void setId(int id) { devId = id; }

    void resetInfo(const af::dim4& dims)
    {
        dim_size = dims;
        dim_strides = calcStrides(dims);
        dim_offsets = af::dim4(0,0,0,0);
    }

    void resetDims(const af::dim4& dims)
    {
        dim_size = dims;
    }

    void modDims(const af::dim4 &newDims);

    void modStrides(const af::dim4 &newStrides);

    bool isEmpty() const;

    bool isScalar() const;

    bool isRow() const;

    bool isColumn() const;

    bool isVector() const;

    bool isComplex() const;

    bool isReal() const;

    bool isDouble() const;

    bool isSingle() const;

    bool isRealFloating() const;

    bool isFloating() const;

    bool isInteger() const;

    bool isBool() const;

    bool isLinear() const;
};

// Returns size and time info for an array object.
// Note this doesn't require template parameters.
const  ArrayInfo&
getInfo(const af_array arr);


af::dim4 toDims(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);

af::dim4 toOffset(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);

af::dim4 toStride(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);
