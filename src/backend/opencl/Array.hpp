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
#include <af/dim4.hpp>
#include <common/ArrayInfo.hpp>
#include <traits.hpp>
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <Param.hpp>
#include <common/jit/Node.hpp>
#include <memory.hpp>
#include <memory>
#include <err_opencl.hpp>

namespace opencl
{
    typedef std::shared_ptr<cl::Buffer> Buffer_ptr;
    using af::dim4;
    template<typename T> class Array;

    template<typename T>
    void evalMultiple(std::vector<Array<T> *> arrays);

    void evalNodes(Param &out, common::Node *node);
    void evalNodes(std::vector<Param> &outputs, std::vector<common::Node *> nodes);

    /// Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createNodeArray(const af::dim4 &size, common::Node_ptr node);

    /// Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createValueArray(const af::dim4 &size, const T& value);

    /// Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> createHostDataArray(const af::dim4 &size, const T * const data);

    template<typename T>
    Array<T> createDeviceDataArray(const af::dim4 &size, const void *data, bool copy = false);

    template<typename T>
    Array<T> createStridedArray(af::dim4 dims, af::dim4 strides, dim_t offset,
                                const T * const in_data, bool is_device) {
        return Array<T>(dims, strides, offset, in_data, is_device);
    }

    /// Copies data to an existing Array object from a host pointer
    template<typename T>
    void writeHostDataArray(Array<T> &arr, const T * const data, const size_t bytes);

    /// Copies data to an existing Array object from a device pointer
    template<typename T>
    void writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes);

    /// Create an Array object and do not assign any values to it.
    /// \NOTE: This object should not be used to initalize an array. Use
    ///       createEmptyArray instead
    template<typename T> Array<T> *initArray();

    /// Creates an empty array of a given size. No data is initialized
    ///
    /// \param[in] size The dimension of the output array
    template<typename T>
    Array<T> createEmptyArray(const af::dim4 &size);

    /// Create an Array object from Param object.
    ///
    /// \param[in] in    The Param array that is created.
    /// \param[in] owner If true, the new Array<T> object is the owner of the data. If false
    ///                  the Array<T> will not delete the object on destruction
    template<typename T>
    Array<T> createParamArray(Param &tmp, bool owner);

    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy=true);

    /// Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    void destroyArray(Array<T> *A);

    template<typename T>
    void *getDevicePtr(const Array<T>& arr)
    {
        const cl::Buffer *buf = arr.device();
        if (!buf) return NULL;
        memLock((T *)buf);
        cl_mem mem = (*buf)();
        return (void *)mem;
    }

    template<typename T>
    void *getRawPtr(const Array<T>& arr)
    {
        const cl::Buffer *buf = arr.get();
        if (!buf) return NULL;
        cl_mem mem = (*buf)();
        return (void *)mem;
    }

    template<typename T>
    class Array
    {
        ArrayInfo info; // This must be the first element of Array<T>
        Buffer_ptr  data;
        af::dim4 data_dims;

        common::Node_ptr node;
        bool ready;
        bool owner;

        Array(af::dim4 dims);

        Array(const Array<T>& parnt, const dim4 &dims, const dim_t &offset, const dim4 &stride);
        Array(Param &tmp, bool owner);
        explicit Array(af::dim4 dims, common::Node_ptr n);
        explicit Array(af::dim4 dims, const T * const in_data);
        explicit Array(af::dim4 dims, cl_mem mem, size_t offset, bool copy);

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

        cl::Buffer* device();
        cl::Buffer* device() const
        {
            return const_cast<Array<T>*>(this)->device();
        }

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

        dim_t getOffset() const
        {
            return info.getOffset();
        }

        Buffer_ptr getData() const
        {
            return data;
        }

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

        operator Param() const
        {
            KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                           {strides()[0], strides()[1], strides()[2], strides()[3]},
                           getOffset()};

            Param out{(cl::Buffer *)this->get(), info};
            return out;
        }

        operator KParam() const
        {
            KParam kinfo = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                            {strides()[0], strides()[1], strides()[2], strides()[3]},
                            getOffset()};

            return kinfo;
        }

        common::Node_ptr getNode() const;
        common::Node_ptr getNode();

    public:
        std::shared_ptr<T> getMappedPtr() const
        {
            auto func = [=] (void* ptr) {
                if(ptr != nullptr) {
                    getQueue().enqueueUnmapMemObject(*data, ptr);
                    ptr = nullptr;
                }
            };

            T *ptr = nullptr;
            if(ptr == nullptr) {
                ptr = (T*)getQueue().enqueueMapBuffer(*const_cast<cl::Buffer*>(get()),
                                                      true, CL_MAP_READ|CL_MAP_WRITE,
                                                      getOffset() * sizeof(T),
                                                      (getDataDims().elements() - getOffset())
                                                      * sizeof(T));
            }

            return std::shared_ptr<T>(ptr, func);
        }


        friend void evalMultiple<T>(std::vector<Array<T> *> arrays);

        friend Array<T> createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T> createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T> createDeviceDataArray<T>(const af::dim4 &size, const void *data, bool copy);
        friend Array<T> createStridedArray<T>(af::dim4 dims, af::dim4 strides, dim_t offset,
                                              const T * const in_data, bool is_device);

        friend Array<T> *initArray<T>();
        friend Array<T> createEmptyArray<T>(const af::dim4 &size);
        friend Array<T> createParamArray<T>(Param &tmp, bool owner);
        friend Array<T> createNodeArray<T>(const af::dim4 &dims, common::Node_ptr node);

        friend Array<T> createSubArray<T>(const Array<T>& parent,
                                          const std::vector<af_seq> &index,
                                          bool copy);

        friend void destroyArray<T>(Array<T> *arr);
        friend void *getDevicePtr<T>(const Array<T>& arr);
        friend void *getRawPtr<T>(const Array<T>& arr);
    };

}
