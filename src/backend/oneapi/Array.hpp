/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <kernel/KParam.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <sycl/buffer.hpp>

#include <nonstd/span.hpp>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

enum class kJITHeuristics;

namespace arrayfire {
namespace common {
template<typename T>
class SparseArray;

class Node;

using Node_ptr = std::shared_ptr<Node>;

}  // namespace common
namespace oneapi {

template<typename T>
struct Param;
template<typename T>
struct AParam;

template<typename T>
using Buffer_ptr = std::shared_ptr<sycl::buffer<T>>;
using af::dim4;
template<typename T>
class Array;

template<typename T>
void evalMultiple(std::vector<Array<T> *> arrays);

template<typename T>
void evalNodes(Param<T> &out, common::Node *node);

template<typename T>
void evalNodes(std::vector<Param<T>> &outputs,
               const std::vector<common::Node *> &nodes);

/// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T> createNodeArray(const af::dim4 &dims, common::Node_ptr node);

/// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T> createValueArray(const af::dim4 &dims, const T &value);

/// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T> createHostDataArray(const af::dim4 &dims, const T *const data);

/// Creates an Array<T> object from a device pointer.
///
/// \param[in] dims The shape of the resulting Array.
/// \param[in] data The device pointer to the data
/// \param[in] copy If true, memory will be allocated and the data will be
///                 copied to the device. If false the data will be used
///                 directly
/// \returns The new Array<T> object based on the device pointer.
template<typename T>
Array<T> createDeviceDataArray(const af::dim4 &dims, void *data,
                               bool copy = false);

template<typename T>
Array<T> createStridedArray(const af::dim4 &dims, const af::dim4 &strides,
                            dim_t offset, const T *const in_data,
                            bool is_device) {
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

/// Create an Array object from Param object.
///
/// \param[in] in    The Param array that is created.
/// \param[in] owner If true, the new Array<T> object is the owner of the data.
/// If false
///                  the Array<T> will not delete the object on destruction
template<typename T>
Array<T> createParamArray(Param<T> &tmp, bool owner);

template<typename T>
Array<T> createSubArray(const Array<T> &parent,
                        const std::vector<af_seq> &index, bool copy = true);

/// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
void destroyArray(Array<T> *A);

/// \brief Checks if the Node can be compiled successfully and the buffers
///        references are not consuming most of the allocated memory
///
/// \param [in] node The root node which needs to be checked
///
/// \returns false if the kernel generated by this node will fail to compile
///          or its nodes are consuming too much memory.
template<typename T>
kJITHeuristics passesJitHeuristics(nonstd::span<common::Node *> node);

template<typename T>
void *getDevicePtr(const Array<T> &arr);

template<typename T>
void *getRawPtr(const Array<T> &arr) {
    // const sycl::buffer<T> *buf = arr.get();
    // if (!buf) return NULL;
    // cl_mem mem = (*buf)();
    // return (void *)mem;

    // TODO:
    return nullptr;
}

template<typename T>
using mapped_ptr = std::unique_ptr<T, std::function<void(void *)>>;

template<typename T>
class Array {
    ArrayInfo info;  // This must be the first element of Array<T>

    /// Pointer to the data
    std::shared_ptr<sycl::buffer<T>> data;

    /// The shape of the underlying parent data.
    af::dim4 data_dims;

    /// Null if this a buffer node. Otherwise this points to a JIT node
    common::Node_ptr node;

    /// If true, the Array object is the parent. If false the data object points
    /// to another array's data
    bool owner;

    Array(const af::dim4 &dims);

    Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset,
          const dim4 &stride);
    Array(Param<T> &tmp, bool owner);
    explicit Array(const af::dim4 &dims, common::Node_ptr n);
    explicit Array(const af::dim4 &dims, const T *const in_data);

    explicit Array(const af::dim4 &dims, sycl::buffer<T> *const mem,
                   size_t offset, bool copy);

    std::shared_ptr<sycl::buffer<T>> getData() const { return data; }

   public:
    Array(const Array<T> &other) = default;

    Array(Array<T> &&other) noexcept = default;

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

    Array(const af::dim4 &dims, const af::dim4 &strides, dim_t offset,
          const T *const in_data, bool is_device = false);
    void resetInfo(const af::dim4 &dims) { info.resetInfo(dims); }
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

    INFO_IS_FUNC(isEmpty);
    INFO_IS_FUNC(isScalar);
    INFO_IS_FUNC(isRow);
    INFO_IS_FUNC(isColumn);
    INFO_IS_FUNC(isVector);
    INFO_IS_FUNC(isComplex);
    INFO_IS_FUNC(isReal);
    INFO_IS_FUNC(isDouble);
    INFO_IS_FUNC(isSingle);
    INFO_IS_FUNC(isHalf);
    INFO_IS_FUNC(isRealFloating);
    INFO_IS_FUNC(isFloating);
    INFO_IS_FUNC(isInteger);
    INFO_IS_FUNC(isBool);
    INFO_IS_FUNC(isLinear);
    INFO_IS_FUNC(isSparse);

#undef INFO_IS_FUNC
    ~Array() = default;

    bool isReady() const { return static_cast<bool>(node) == false; }
    bool isOwner() const { return owner; }

    void eval();
    void eval() const;

    sycl::buffer<T> *device();
    sycl::buffer<T> *device() const {
        return const_cast<Array<T> *>(this)->device();
    }

    sycl::buffer<T> *get() const {
        if (!isReady()) eval();
        return data.get();
    }

    int useCount() const { return data.use_count(); }

    dim_t getOffset() const { return info.getOffset(); }

    dim4 getDataDims() const { return data_dims; }

    void setDataDims(const dim4 &new_dims);

    size_t getAllocatedBytes() const;

    operator Param<T>() const {
        KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                       {strides()[0], strides()[1], strides()[2], strides()[3]},
                       getOffset()};

        Param<T> out{(sycl::buffer<T> *)this->get(), info};
        return out;
    }

    operator AParam<T>() {
        AParam<T> out(*getData(), dims().get(), strides().get(), getOffset());
        return out;
    }

    operator KParam() const {
        KParam kinfo = {
            {dims()[0], dims()[1], dims()[2], dims()[3]},
            {strides()[0], strides()[1], strides()[2], strides()[3]},
            getOffset()};

        return kinfo;
    }

    common::Node_ptr getNode() const;
    common::Node_ptr getNode();

   public:
    mapped_ptr<T> getMappedPtr(cl_map_flags map_flags = CL_MAP_READ |
                                                        CL_MAP_WRITE) const {
        if (!isReady()) eval();
        auto func = [data = data](void *ptr) {
            if (ptr != nullptr) {
                // cl_int err = getQueue().enqueueUnmapMemObject(*data, ptr);
                // UNUSED(err);
                ptr = nullptr;
            }
        };

        // T *ptr = (T *)getQueue().enqueueMapBuffer(
        //*static_cast<const sycl::buffer<T> *>(get()), CL_TRUE, map_flags,
        // getOffset() * sizeof(T), elements() * sizeof(T), nullptr, nullptr,
        // nullptr);

        return mapped_ptr<T>(nullptr, func);
    }

    friend void evalMultiple<T>(std::vector<Array<T> *> arrays);

    friend Array<T> createValueArray<T>(const af::dim4 &dims, const T &value);
    friend Array<T> createHostDataArray<T>(const af::dim4 &dims,
                                           const T *const data);
    friend Array<T> createDeviceDataArray<T>(const af::dim4 &dims, void *data,
                                             bool copy);
    friend Array<T> createStridedArray<T>(const af::dim4 &dims,
                                          const af::dim4 &strides, dim_t offset,
                                          const T *const in_data,
                                          bool is_device);

    friend Array<T> createEmptyArray<T>(const af::dim4 &dims);
    friend Array<T> createParamArray<T>(Param<T> &tmp, bool owner);
    friend Array<T> createNodeArray<T>(const af::dim4 &dims,
                                       common::Node_ptr node);

    friend Array<T> createSubArray<T>(const Array<T> &parent,
                                      const std::vector<af_seq> &index,
                                      bool copy);

    friend void destroyArray<T>(Array<T> *arr);
    friend void *getDevicePtr<T>(const Array<T> &arr);
    friend void *getRawPtr<T>(const Array<T> &arr);
};

}  // namespace oneapi
}  // namespace arrayfire
