/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

//This is the array implementation class.
#pragma once
#include <af/dim4.hpp>
#include <common/ArrayInfo.hpp>
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <TNJ/Node.hpp>
#include <Param.hpp>
#include <memory.hpp>
#include <memory>
#include <algorithm>
#include <vector>
#include <platform.hpp>
#include <queue.hpp>

// cpu::Array class forward declaration
namespace cpu
{
template<typename T> class Array;
// kernel::evalArray fn forward declaration
namespace kernel
{
    template<typename T> void evalArray(Param<T> in, TNJ::Node_ptr node);

    template<typename T>
    void evalMultiple(std::vector<Param<T>> arrays, std::vector<TNJ::Node_ptr> nodes);

}
}

namespace cpu
{

    using std::shared_ptr;
    using af::dim4;

    template<typename T>
    void evalMultiple(std::vector<Array<T> *> arrays);

    template<typename T> class Array;

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createNodeArray(const af::dim4 &size, TNJ::Node_ptr node);

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
        getQueue().sync();
        return (void *)(arr.get(false));
    }

    // Array Array Implementation
    template<typename T>
    class Array
    {
        ArrayInfo info; // Must be the first element of Array<T>
        //TODO: Generator based array

        //data if parent. empty if child
        std::shared_ptr<T> data;
        af::dim4 data_dims;
        TNJ::Node_ptr node;

        bool ready;
        bool owner;

        Array() = default;
        Array(dim4 dims);

        explicit Array(dim4 dims, const T * const in_data, bool is_device, bool copy_device=false);
        Array(const Array<T>& parnt, const dim4 &dims, const dim_t &offset, const dim4 &stride);
        explicit Array(af::dim4 dims, TNJ::Node_ptr n);

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

        ~Array() = default;

        bool isReady() const { return ready; }

        bool isOwner() const { return owner; }

        void eval();
        void eval() const;

        dim_t getOffset() const { return info.getOffset(); }
        shared_ptr<T> getData() const {return data; }

        dim4 getDataDims() const
        {
            return data_dims;
        }

        void setDataDims(const dim4 &new_dims)
        {
            modDims(new_dims);
            data_dims = new_dims;
        }

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
            return const_cast<T*>(static_cast<const Array<T>*>(this)->get(withOffset));
        }

        const T* get(bool withOffset = true) const
        {
            if (!data.get()) eval();
            return data.get() + (withOffset ? getOffset() : 0);
        }

        int useCount() const
        {
            if (!data.get()) eval();
            return data.use_count();
        }

        operator Param<T>()
        {
            return Param<T>(this->get(), this->dims(), this->strides());
        }

        operator CParam<T>() const
        {
            return CParam<T>(this->get(), this->dims(), this->strides());
        }

        TNJ::Node_ptr getNode() const;

        friend void evalMultiple<T>(std::vector<Array<T> *> arrays);

        friend Array<T> createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T> createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T> createDeviceDataArray<T>(const af::dim4 &size, const void *data);

        friend Array<T> *initArray<T>();
        friend Array<T> createEmptyArray<T>(const af::dim4 &size);
        friend Array<T> createNodeArray<T>(const af::dim4 &dims, TNJ::Node_ptr node);

        friend Array<T> createSubArray<T>(const Array<T>& parent,
                                          const std::vector<af_seq> &index,
                                          bool copy);

        friend void kernel::evalArray<T>(Param<T> in, TNJ::Node_ptr node);
        friend void kernel::evalMultiple<T>(std::vector<Param<T>> arrays,
                                            std::vector<TNJ::Node_ptr> nodes);

        friend void destroyArray<T>(Array<T> *arr);
        friend void *getDevicePtr<T>(const Array<T>& arr);
        friend void *getRawPtr<T>(const Array<T>& arr);
    };

}
