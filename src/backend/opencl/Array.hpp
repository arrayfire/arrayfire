/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <platform.hpp>
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <traits.hpp>
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <Param.hpp>
#include <JIT/Node.hpp>
#include <memory.hpp>
#include <memory>

namespace opencl
{
    using af::dim4;
    typedef std::shared_ptr<cl::Buffer> Buffer_ptr;

    template<typename T> class Array;

    void evalNodes(Param &out, JIT::Node *node);

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

    // Create an Array object from Param
    template<typename T>
    Array<T> createParamArray(Param &tmp);

    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy=true);

    template<typename T>
    void evalArray(const Array<T> &A);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    void destroyArray(Array<T> *A);

    template<typename T>
    void *getDevicePtr(const Array<T>& arr)
    {
        memPop((T *)arr.get());
        return (void *)((*arr.get())());
    }

    template<typename T>
    class Array
    {
        ArrayInfo info; // This must be the first element of Array<T>
        Buffer_ptr  data;
        af::dim4 data_dims;

        JIT::Node_ptr node;
        dim_t offset;
        bool ready;
        bool owner;

        Array(af::dim4 dims);
        Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
        Array(Param &tmp);
        explicit Array(af::dim4 dims, JIT::Node_ptr n);
        explicit Array(af::dim4 dims, const T * const in_data);
        explicit Array(af::dim4 dims, cl_mem mem);

    public:

        void resetInfo(const af::dim4& dims)        { info.resetInfo(dims);         }
        void resetDims(const af::dim4& dims)        { info.resetDims(dims);         }
        void modDims(const af::dim4 &newDims)       { info.modDims(newDims);        }
        void modStrides(const af::dim4 &newStrides) { info.modStrides(newStrides);  }
        void setId(int id)                          { info.setId(id);               }

#define INFO_FUNC(RET_TYPE, NAME)   \
    RET_TYPE NAME() const { return info.NAME(); }

        INFO_FUNC(const af_dtype& ,getType)
        INFO_FUNC(const af::dim4& ,offsets)
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

#undef INFO_IS_FUNC
        ~Array();

        bool isReady() const { return ready; }
        bool isOwner() const { return owner; }

        void eval();
        void eval() const;

        //FIXME: This should do a copy if it is not owner. You do not want to overwrite parents data
        cl::Buffer *get()
        {
            if (!isReady()) eval();
            return data.get();
        }

        const cl::Buffer *get() const
        {
            if (!isReady()) eval();
            return data.get();
        }

        int useCount() const
        {
            if (!isReady()) eval();
            return data.use_count();
        }

        const dim_t getOffset() const
        {
            return offset;
        }

        Buffer_ptr getData() const
        {
            return data;
        }

        dim4 getDataDims() const
        {
            // This is for moddims
            // dims and data_dims are different when moddims is used
            return isOwner() ? dims() : data_dims;
        }

        operator Param() const
        {
            KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                           {strides()[0], strides()[1], strides()[2], strides()[3]},
                           getOffset()};

            Param out = {(cl::Buffer *)this->get(), info};
            return out;
        }

        JIT::Node_ptr getNode() const;

        friend Array<T> createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T> createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T> createDeviceDataArray<T>(const af::dim4 &size, const void *data);

        friend Array<T> *initArray<T>();
        friend Array<T> createEmptyArray<T>(const af::dim4 &size);
        friend Array<T> createParamArray<T>(Param &tmp);
        friend Array<T> createNodeArray<T>(const af::dim4 &dims, JIT::Node_ptr node);

        friend Array<T> createSubArray<T>(const Array<T>& parent,
                                          const std::vector<af_seq> &index,
                                          bool copy);

        friend void destroyArray<T>(Array<T> *arr);
        friend void evalArray<T>(const Array<T> &arr);
        friend void *getDevicePtr<T>(const Array<T>& arr);
    };

}
