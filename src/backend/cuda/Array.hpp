/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/dim4.hpp>
#include <common/ArrayInfo.hpp>
#include "traits.hpp"
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <Param.hpp>
#include <JIT/Node.hpp>
#include <vector>
#include <memory.hpp>

namespace cuda
{
    using af::dim4;

    template<typename T> class Array;

    template<typename T>
    void evalNodes(Param<T> &out, JIT::Node *node);

    template<typename T>
    void evalNodes(std::vector<Param<T> > &out, std::vector<JIT::Node *> nodes);

    template<typename T>
    void evalMultiple(std::vector<Array<T> *> arrays);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createNodeArray(const af::dim4 &size, JIT::Node_ptr node);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createValueArray(const af::dim4 &size, const T& value);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createHostDataArray(const af::dim4 &size, const T * const data);

    template<typename T>
    Array<T> createDeviceDataArray(const af::dim4 &size, const void *data);

    // Copies data to an existing Array object from a host pointer
    template<typename T>
    void writeHostDataArray(Array<T> &arr, const T * const data, const size_t bytes);

    // Copies data to an existing Array object from a device pointer
    template<typename T>
    void writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes);

    // Create an Array object and do not assign any values to it
    template<typename T> Array<T> *initArray();

    template<typename T>
    Array<T> createEmptyArray(const af::dim4 &size);

    // Create an Array object from Param<T>
    template<typename T>
    Array<T> createParamArray(Param<T> &tmp, bool owner);

    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy=true);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    void destroyArray(Array<T> *A);

    template<typename T>
    void *getDevicePtr(const Array<T>& arr)
    {
        T *ptr = arr.device();
        memLock(ptr);
        return (void *)ptr;
    }

    template<typename T>
    void *getRawPtr(const Array<T>& arr)
    {
        return (void *)(arr.get(false));
    }

    template<typename T>
    class Array
    {
        ArrayInfo       info; // This must be the first element of Array<T>
        std::shared_ptr<T> data;
        af::dim4 data_dims;

        JIT::Node_ptr node;
        bool ready;
        bool owner;

        Array(af::dim4 dims);

        explicit Array(af::dim4 dims, const T * const in_data, bool is_device = false, bool copy_device = false);
        Array(const Array<T>& parnt, const dim4 &dims, const dim_t &offset, const dim4 &stride);
        Array(Param<T> &tmp, bool owner);
        Array(af::dim4 dims, JIT::Node_ptr n);
    public:

        Array(af::dim4 dims, af::dim4 strides, dim_t offset,
              const T * const in_data, bool is_device = false);

        void resetInfo(const af::dim4& dims)        { info.resetInfo(dims);         }
        void resetDims(const af::dim4& dims)        { info.resetDims(dims);         }
        void modDims(const af::dim4 &newDims)       { info.modDims(newDims);        }
        void modStrides(const af::dim4 &newStrides) { info.modStrides(newStrides);  }
        void setId(int id)                          { info.setId(id);               }

#define INFO_FUNC(RET_TYPE, NAME)   \
    RET_TYPE NAME() const { return info.NAME(); }

        INFO_FUNC(const af_dtype& ,getType)
        INFO_FUNC(const af::dim4& ,strides)
        INFO_FUNC(size_t          ,elements)
        INFO_FUNC(size_t          ,ndims)
        INFO_FUNC(const af::dim4& ,dims )
        INFO_FUNC(int             ,getDevId)

#undef INFO_FUNC

#define INFO_IS_FUNC(NAME)\
    bool NAME () const { return info.NAME(); }

        INFO_IS_FUNC(isEmpty);
        INFO_IS_FUNC(isScalar);
        INFO_IS_FUNC(isRow);
        INFO_IS_FUNC(isColumn);
        INFO_IS_FUNC(isVector);
        INFO_IS_FUNC(isComplex);
        INFO_IS_FUNC(isReal);
        INFO_IS_FUNC(isDouble);
        INFO_IS_FUNC(isSingle);
        INFO_IS_FUNC(isRealFloating);
        INFO_IS_FUNC(isFloating);
        INFO_IS_FUNC(isInteger);
        INFO_IS_FUNC(isBool);
        INFO_IS_FUNC(isLinear);
        INFO_IS_FUNC(isSparse);

#undef INFO_IS_FUNC

        ~Array();

        bool isReady() const { return ready; }
        bool isOwner() const { return owner; }

        void eval();
        void eval() const;

        dim_t getOffset() const { return info.getOffset(); }
        std::shared_ptr<T> getData() const { return data; }

        dim4 getDataDims() const
        {
            return data_dims;
        }

        void setDataDims(const dim4 &new_dims);

        size_t getAllocatedBytes() const
        {
            if (!isReady()) return 0;
            size_t bytes = memoryManager().allocated(data.get());
            // External device poitner
            if (bytes == 0 && data.get()) {
                return data_dims.elements() * sizeof(T);
            }
            return  bytes;
        }

        T* device();

        T* device() const
        {
            return const_cast<Array<T>*>(this)->device();
        }

        T* get(bool withOffset = true)
        {
            if (!isReady()) eval();
            return const_cast<T*>(static_cast<const Array<T>*>(this)->get(withOffset));
        }

        //FIXME: implement withOffset parameter
        const   T* get(bool withOffset = true) const
        {
            if (!isReady()) eval();
            return data.get() + (withOffset ? getOffset() : 0);
        }

        int useCount() const
        {
            if (!isReady()) eval();
            return data.use_count();
        }

        operator Param<T>()
        {
            return Param<T>(this->get(), this->dims().get(), this->strides().get());
        }

        operator CParam<T>() const
        {
            return CParam<T>(this->get(), this->dims().get(), this->strides().get());
        }

        JIT::Node_ptr getNode();
        JIT::Node_ptr getNode() const;

        friend void evalMultiple<T>(std::vector<Array<T> *> arrays);
        friend Array<T> createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T> createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T> createDeviceDataArray<T>(const af::dim4 &size, const void *data);

        friend Array<T> *initArray<T>();
        friend Array<T> createEmptyArray<T>(const af::dim4 &size);
        friend Array<T> createParamArray<T>(Param<T> &tmp, bool owner);
        friend Array<T> createNodeArray<T>(const af::dim4 &dims, JIT::Node_ptr node);

        friend Array<T> createSubArray<T>(const Array<T>& parent,
                                          const std::vector<af_seq> &index,
                                          bool copy);

        friend void destroyArray<T>(Array<T> *arr);
        friend void *getDevicePtr<T>(const Array<T>& arr);
        friend void *getRawPtr<T>(const Array<T>& arr);
    };

}
