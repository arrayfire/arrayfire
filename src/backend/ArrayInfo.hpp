/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <defines.hpp>
#include <af/array.h>
#include <af/util.h>
#include <af/dim4.hpp>
#include <af/device.h>
#include <vector>
#include <cstddef>

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
    // The devId variable stores information about the deviceId as well as the backend.
    // The 4 LSBs (0-3) are used to store the device ID.
    // The 4th LSB is set to 1 if backend is CPU
    // The 5th LSB is set to 1 if backend is CUDA
    // The 6th LSB is set to 1 if backend is OpenCL
    // This information can be retrieved directly from an af_array by doing
    //     int* devId = reinterpret_cast<int*>(a); // a is an af_array
    //     af_backend backendID = *devId >> 3;  // Returns 1, 2, 4 for CPU, CUDA or OpenCL respectively
    //     int        deviceID  = *devId & 0xf; // Returns devices ID between 0-15
    // This is possible by doing a static_assert on devId
    //
    // This can be changed in the future if the need arises for more devices as this
    // implementation is internal. Make sure to change the bit shift ops when
    // such a change is being made
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
    {
        af_init();
        setId(id);
#if __cplusplus > 199711l
    static_assert(offsetof(ArrayInfo, devId) == 0,
                  "ArrayInfo::devId must be the first member variable of ArrayInfo. \
                   devId is used to encode the backend into the integer. \
                   This is then used in the unified backend to check mismatched arrays.");
#endif
    }

#if __cplusplus > 199711L
    //Copy constructors are deprecated if there is a
    //user-defined destructor in c++11
    ArrayInfo() = default;
    ArrayInfo(const ArrayInfo& other) = default;
#endif

    const af_dtype& getType() const     { return type;                  }

    const af::dim4& offsets() const     { return dim_offsets;           }

    const af::dim4& strides() const     { return dim_strides;           }

    size_t elements() const             { return dim_size.elements();   }
    size_t ndims() const                { return dim_size.ndims();      }
    const af::dim4& dims() const        { return dim_size;              }

    int getDevId() const;

    void setId(int id) const;

    void setId(int id);

    af_backend getBackendId() const;

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
#if __cplusplus > 199711l
    static_assert(std::is_standard_layout<ArrayInfo>::value, "ArrayInfo must be a standard layout type");
#endif

// Returns size and time info for an array object.
// Note this doesn't require template parameters.
const  ArrayInfo&
getInfo(const af_array arr);


af::dim4 toDims(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);

af::dim4 toOffset(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);

af::dim4 toStride(const std::vector<af_seq>& seqs, const af::dim4 &parentDims);
