/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/defines.hpp>
#include <af/device.h>
#include <af/dim4.hpp>
#include <cstddef>
#include <vector>

af::dim4 calcStrides(const af::dim4& parentDim);

af::dim4 getOutDims(const af::dim4& ldims, const af::dim4& rdims,
                    bool batchMode);

/// Array Arrayementation Info class
// This class is the base class to all Array objects. The purpose of this class
// was to have a way to retrieve basic information of an Array object without
// specifying what type the object is at compile time.
class ArrayInfo {
   private:
    // The devId variable stores information about the deviceId as well as the
    // backend. The 8 LSBs (0-7) are used to store the device ID. The 09th LSB
    // is set to 1 if backend is CPU The 10th LSB is set to 1 if backend is CUDA
    // The 11th LSB is set to 1 if backend is OpenCL The 12th LSB is set to 1
    // for oneAPI
    // This information can be retrieved directly from an af_array by doing
    //     int* devId = reinterpret_cast<int*>(a); // a is an af_array
    //     af_backend backendID = *devId >> 8;   // Returns 1, 2, 4 for CPU,
    //     CUDA or OpenCL respectively int        deviceID  = *devId & 0xff; //
    //     Returns devices ID between 0-255
    // This is possible by doing a static_assert on devId
    //
    // This can be changed in the future if the need arises for more devices as
    // this implementation is internal. Make sure to change the bit shift ops
    // when such a change is being made
    unsigned devId;
    af_dtype type;
    af::dim4 dim_size;
    dim_t offset;
    af::dim4 dim_strides;
    bool is_sparse;

   public:
    ArrayInfo(unsigned id, af::dim4 size, dim_t offset_, af::dim4 stride,
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
    }

    ArrayInfo(unsigned id, af::dim4 size, dim_t offset_, af::dim4 stride,
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

    ArrayInfo()                       = default;
    ArrayInfo(const ArrayInfo& other) = default;
    ArrayInfo(ArrayInfo&& other)      = default;

    ArrayInfo& operator=(ArrayInfo other) noexcept {
        swap(other);
        return *this;
    }

    void swap(ArrayInfo& other) noexcept {
        using std::swap;
        swap(devId, other.devId);
        swap(type, other.type);
        swap(dim_size, other.dim_size);
        swap(offset, other.offset);
        swap(dim_strides, other.dim_strides);
        swap(is_sparse, other.is_sparse);
    }

    const af_dtype& getType() const { return type; }

    dim_t getOffset() const { return offset; }

    const af::dim4& strides() const { return dim_strides; }

    dim_t elements() const { return dim_size.elements(); }
    dim_t ndims() const { return dim_size.ndims(); }
    const af::dim4& dims() const { return dim_size; }
    size_t total() const { return offset + dim_strides[3] * dim_size[3]; }

    unsigned getDevId() const;

    void setId(int id) const;

    void setId(int id);

    af_backend getBackendId() const;

    void resetInfo(const af::dim4& dims) {
        dim_size    = dims;
        dim_strides = calcStrides(dims);
        offset      = 0;
    }

    void resetDims(const af::dim4& dims) { dim_size = dims; }

    void modDims(const af::dim4& newDims);

    void modStrides(const af::dim4& newStrides);

    bool isEmpty() const;

    bool isScalar() const;

    bool isRow() const;

    bool isColumn() const;

    bool isVector() const;

    bool isComplex() const;

    bool isReal() const;

    bool isDouble() const;

    bool isSingle() const;

    bool isHalf() const;

    bool isRealFloating() const;

    bool isFloating() const;

    bool isInteger() const;

    bool isBool() const;

    bool isLinear() const;

    bool isSparse() const;
};
static_assert(std::is_standard_layout<ArrayInfo>::value,
              "ArrayInfo must be a standard layout type");

af::dim4 toDims(const std::vector<af_seq>& seqs, const af::dim4& parentDims);

af::dim4 toOffset(const std::vector<af_seq>& seqs, const af::dim4& parentDims);

af::dim4 toStride(const std::vector<af_seq>& seqs, const af::dim4& parentDims);
