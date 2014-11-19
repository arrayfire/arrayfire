/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Workaround for BOOST_NOINLINE not being defined with nvcc / CUDA < 6.5
#if CUDA_VERSION < 6050
#define BOOST_NOINLINE __attribute__ ((noinline))
#endif

#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include "traits.hpp"
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <cuda_runtime_api.h>
#include <Param.hpp>
#include <JIT/Node.hpp>
#include <boost/shared_ptr.hpp>

namespace cuda
{

    using af::dim4;
    using boost::shared_ptr;

    template<typename T> class Array;

    template<typename T>
    void evalNodes(Param<T> &out, JIT::Node *node);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createNodeArray(const af::dim4 &size, JIT::Node_ptr node);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createValueArray(const af::dim4 &size, const T& value);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createHostDataArray(const af::dim4 &size, const T * const data);

    template<typename T>
    Array<T>*
    createDeviceDataArray(const af::dim4 &size, const void *data);

    // Create an Array object and do not assign any values to it
    template<typename T>
    Array<T>*
    createEmptyArray(const af::dim4 &size);

    // Create an Array object from Param<T>
    template<typename T>
    Array<T>*
    createParamArray(Param<T> &tmp);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    void
    destroyArray(Array<T> &A);

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const af::dim4 &dims, const af::dim4 &offset, const af::dim4 &stride);

    // Creates a pure reference Array - a virtual view, no copies are made
    template<typename T>
    Array<T> *
    createRefArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor=1.0);

    template<typename T>
    T* cudaMallocWrapper(const size_t &elements);

    template<typename T>
    void *getDevicePtr(const Array<T>& arr)
    {
        return (void *)arr.get();
    }

    template<typename T>
    class Array : public ArrayInfo
    {
        shared_ptr<T> data;
        const Array*  parent;

        JIT::Node_ptr node;
        bool ready;

        Array(af::dim4 dims);
        explicit Array(af::dim4 dims, const T * const in_data, bool is_device = false);
        Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
        Array(Param<T> &tmp);
        Array(af::dim4 dims, JIT::Node_ptr n);

    public:

        ~Array();

        bool isReady() const { return ready; }
        bool isOwner() const { return parent == NULL; }

        void eval();
        void eval() const;

        //FIXME: This should do a copy if it is not owner. You do not want to overwrite parents data
        T* get(bool withOffset = true)
        {
            if (!isReady()) eval();
            return const_cast<T*>(static_cast<const Array<T>*>(this)->get(withOffset));
        }

        //FIXME: implement withOffset parameter
        const   T* get(bool withOffset = true) const
        {
            if (!isReady()) eval();
            if (isOwner()) return data.get();
            return parent->data.get() + (withOffset ? calcOffset(parent->strides(), this->offsets()) : 0);
        }

        operator Param<T>()
        {
            Param<T> out;
            out.ptr = this->get();
            for (int  i = 0; i < 4; i++) {
                out.dims[i] = dims()[i];
                out.strides[i] = strides()[i];
            }
            return out;
        }

        operator CParam<T>() const
        {
            CParam<T> out(this->get(), this->dims().get(), this->strides().get());
            return out;
        }

        JIT::Node_ptr getNode() const;

        friend Array<T>* createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T>* createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T>* createDeviceDataArray<T>(const af::dim4 &size, const void *data);

        friend Array<T>* createEmptyArray<T>(const af::dim4 &size);
        friend Array<T>* createParamArray<T>(Param<T> &tmp);
        friend Array<T>* createNodeArray<T>(const af::dim4 &dims, JIT::Node_ptr node);
        friend Array<T>* createSubArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend Array<T>* createRefArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend void      destroyArray<T>(Array<T> &arr);
        friend void *getDevicePtr<T>(const Array<T>& arr);
    };

}
