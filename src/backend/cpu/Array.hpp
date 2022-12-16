/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// This is the array implementation class.
#pragma once

#include <Param.hpp>
#include <common/ArrayInfo.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/jit/Node.hpp>
#include <jit/Node.hpp>
#include <memory.hpp>
#include <platform.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>

#include <nonstd/span.hpp>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace arrayfire {
namespace cpu {

namespace jit {
template<typename T>
class BufferNode;
}

namespace kernel {
template<typename T>
void evalArray(Param<T> in, common::Node_ptr node);

template<typename T>
void evalMultiple(std::vector<Param<T>> arrays,
                  std::vector<common::Node_ptr> nodes);

}  // namespace kernel

template<typename T>
class Array;

using af::dim4;
using std::shared_ptr;

template<typename T>
void evalMultiple(std::vector<Array<T> *> array_ptrs);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T> createNodeArray(const af::dim4 &dims, common::Node_ptr node);

template<typename T>
Array<T> createValueArray(const af::dim4 &dims, const T &value);

// Creates an array and copies from the \p data pointer located in host memory
//
// \param[in] dims The dimension of the array
// \param[in] data The data that will be copied to the array
template<typename T>
Array<T> createHostDataArray(const af::dim4 &dims, const T *const data);

template<typename T>
Array<T> createDeviceDataArray(const af::dim4 &dims, void *data);

template<typename T>
Array<T> createStridedArray(af::dim4 dims, af::dim4 strides, dim_t offset,
                            T *const in_data, bool is_device) {
    return Array<T>(dims, strides, offset, in_data, is_device);
}

/// Copies data to an existing Array object from a host pointer
template<typename T>
void writeHostDataArray(Array<T> &arr, const T *const data, const size_t bytes);

/// Copies data to an existing Array object from a device pointer
template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes);

/// Creates an empty array of a given size. No data is initialized
///
/// \param[in] size The dimension of the output array
template<typename T>
Array<T> createEmptyArray(const af::dim4 &dims);

template<typename T>
Array<T> createSubArray(const Array<T> &parent,
                        const std::vector<af_seq> &index, bool copy = true);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
void destroyArray(Array<T> *A);

template<typename T>
kJITHeuristics passesJitHeuristics(nonstd::span<common::Node *> node);

template<typename T>
void *getDevicePtr(const Array<T> &arr) {
    T *ptr = arr.device();
    memLock(ptr);

    return (void *)ptr;
}

template<typename T>
void *getRawPtr(const Array<T> &arr) {
    getQueue().sync();
    return (void *)(arr.get(false));
}

// Array Array Implementation
template<typename T>
class Array {
    ArrayInfo info;  // Must be the first element of Array<T>

    /// Pointer to the data
    std::shared_ptr<T> data;

    /// The shape of the underlying parent data.
    af::dim4 data_dims;

    /// Null if this a buffer node. Otherwise this points to a JIT node
    common::Node_ptr node;

    /// If true, the Array object is the parent. If false the data object points
    /// to another array's data
    bool owner;

    /// Default constructor
    Array() = default;

    /// Creates an uninitialized array of a specific shape
    Array(dim4 dims);

    explicit Array(const af::dim4 &dims, T *const in_data, bool is_device,
                   bool copy_device = false);
    Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset,
          const dim4 &stride);
    explicit Array(const af::dim4 &dims, common::Node_ptr n);
    Array(const af::dim4 &dims, const af::dim4 &strides, dim_t offset,
          T *const in_data, bool is_device = false);

   public:
    Array<T>(const Array<T> &other) = default;
    Array<T>(Array<T> &&other)      = default;

    Array<T> &operator=(Array<T> other) noexcept {
        swap(other);
        return *this;
    }

    void swap(Array<T> &other) noexcept {
        using std::swap;
        swap(info, other.info);
        swap(data, other.data);
        swap(data_dims, other.data_dims);
        swap(node, other.node);
        swap(owner, other.owner);
    }

    void resetInfo(const af::dim4 &dims) { info.resetInfo(dims); }

    // Modifies the dimensions of the array without modifing the underlying
    // data
    void resetDims(const af::dim4 &dims) { info.resetDims(dims); }
    void modDims(const af::dim4 &newDims) { info.modDims(newDims); }
    void modStrides(const af::dim4 &newStrides) { info.modStrides(newStrides); }
    void setId(int id) { info.setId(id); }

#define INFO_FUNC(RET_TYPE, NAME) \
    RET_TYPE NAME() const { return info.NAME(); }

    INFO_FUNC(const af_dtype &, getType)
    INFO_FUNC(const af::dim4 &, strides)
    INFO_FUNC(dim_t, elements)
    INFO_FUNC(dim_t, ndims)
    INFO_FUNC(const af::dim4 &, dims)
    INFO_FUNC(int, getDevId)

#undef INFO_FUNC

#define INFO_IS_FUNC(NAME) \
    bool NAME() const { return info.NAME(); }

    INFO_IS_FUNC(isEmpty)
    INFO_IS_FUNC(isScalar)
    INFO_IS_FUNC(isRow)
    INFO_IS_FUNC(isColumn)
    INFO_IS_FUNC(isVector)
    INFO_IS_FUNC(isComplex)
    INFO_IS_FUNC(isReal)
    INFO_IS_FUNC(isDouble)
    INFO_IS_FUNC(isSingle)
    INFO_IS_FUNC(isHalf);
    INFO_IS_FUNC(isRealFloating)
    INFO_IS_FUNC(isFloating)
    INFO_IS_FUNC(isInteger)
    INFO_IS_FUNC(isBool)
    INFO_IS_FUNC(isLinear)
    INFO_IS_FUNC(isSparse)

#undef INFO_IS_FUNC

    ~Array() = default;

    bool isReady() const { return static_cast<bool>(node) == false; }

    bool isOwner() const { return owner; }

    void eval();
    void eval() const;

    dim_t getOffset() const { return info.getOffset(); }
    shared_ptr<T> getData() const { return data; }

    dim4 getDataDims() const { return data_dims; }

    void setDataDims(const dim4 &new_dims);

    size_t getAllocatedBytes() const {
        if (!isReady()) return 0;
        size_t bytes = memoryManager().allocated(data.get());
        // External device poitner
        if (bytes == 0 && data.get()) {
            return data_dims.elements() * sizeof(T);
        }
        return bytes;
    }

    T *device();

    T *device() const { return const_cast<Array<T> *>(this)->device(); }

    T *get(bool withOffset = true) {
        return const_cast<T *>(
            static_cast<const Array<T> *>(this)->get(withOffset));
    }

    const T *get(bool withOffset = true) const {
        if (!data.get()) eval();
        return data.get() + (withOffset ? getOffset() : 0);
    }

    int useCount() const { return static_cast<int>(data.use_count()); }

    operator Param<T>() {
        return Param<T>(this->get(), this->dims(), this->strides());
    }

    operator CParam<T>() const {
        return CParam<T>(this->get(), this->dims(), this->strides());
    }

    common::Node_ptr getNode() const;
    common::Node_ptr getNode();

    friend void evalMultiple<T>(std::vector<Array<T> *> arrays);

    friend Array<T> createValueArray<T>(const af::dim4 &dims, const T &value);
    friend Array<T> createHostDataArray<T>(const af::dim4 &dims,
                                           const T *const data);
    friend Array<T> createDeviceDataArray<T>(const af::dim4 &dims, void *data);
    friend Array<T> createStridedArray<T>(af::dim4 dims, af::dim4 strides,
                                          dim_t offset, T *const in_data,
                                          bool is_device);

    friend Array<T> createEmptyArray<T>(const af::dim4 &dims);
    friend Array<T> createNodeArray<T>(const af::dim4 &dims,
                                       common::Node_ptr node);

    friend Array<T> createSubArray<T>(const Array<T> &parent,
                                      const std::vector<af_seq> &index,
                                      bool copy);

    friend void kernel::evalArray<T>(Param<T> in, common::Node_ptr node);
    friend void kernel::evalMultiple<T>(std::vector<Param<T>> arrays,
                                        std::vector<common::Node_ptr> nodes);

    friend void destroyArray<T>(Array<T> *arr);
    friend void *getDevicePtr<T>(const Array<T> &arr);
    friend void *getRawPtr<T>(const Array<T> &arr);
};

}  // namespace cpu
}  // namespace arrayfire
