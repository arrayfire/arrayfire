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
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>
#include <TNJ/Node.hpp>
#include <memory>
#include <algorithm>

namespace cpu
{

    using std::shared_ptr;
    using af::dim4;

    template<typename T> class Array;

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createNodeArray(const af::dim4 &size, TNJ::Node *node);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T> *
    createValueArray(const af::dim4 &size, const T& value);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createHostDataArray(const af::dim4 &size, const T * const data);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createDeviceDataArray(const af::dim4 &size, const void *data);

    // Create an Array object and do not assign any values to it
    template<typename T>
    Array<T>*
    createEmptyArray(const af::dim4 &size);

    // Creates a new Array View(sub array).
    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

    // Creates a pure reference Array - a virtual view, no copies are made
    template<typename T>
    Array<T> *
    createRefArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value=outType(0));

    template<typename T>
    void scaleArray(Array<T> &arr, double factor);

    template<typename T>
    void
    destroyArray(Array<T> &arr);

    template<typename T>
    void
    operator <<(std::ostream &out, const Array<T> &arr);

    template<typename T>
    void *getDevicePtr(const Array<T>& arr)
    {
        return (void *)arr.get();
    }

    // Array Array Implementation
    template<typename T>
    class Array : public ArrayInfo
    {
        //TODO: Generator based array

        //data if parent. empty if child
        std::shared_ptr<T> data;

        //If parent is valid. use offset to get values
        const Array<T> *parent;

        TNJ::Node *node;
        bool ready;

        Array(dim4 dims);
        explicit Array(dim4 dims, const T * const in_data);
        Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
        explicit Array(af::dim4 dims, TNJ::Node *n);

    public:

        ~Array();

        bool isReady() const { return ready; }

        bool isOwner() const
        {
            return parent == nullptr;
        }

        void eval();
        void eval() const;

        //FIXME: This should do a copy if it is not owner. You do not want to overwrite parents data
        T* get(bool withOffset = true)
        {
            return const_cast<T*>(static_cast<const Array<T>*>(this)->get(withOffset));
        }

        const T* get(bool withOffset = true) const
        {

            if (!isReady()) eval();

            const T* ptr = nullptr;
            if(isOwner()) {
                ptr = data.get();
            } else {
                size_t offset = 0;
                if(withOffset) {
                    offset = calcOffset(parent->strides(), this->offsets());
                }
                ptr = parent->data.get() + offset;
            }
            return ptr;
        }

        TNJ::Node *getNode() const;

        friend Array<T>* createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T>* createHostDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T>* createDeviceDataArray<T>(const af::dim4 &size, const void *data);
        friend Array<T>* createEmptyArray<T>(const af::dim4 &size);
        friend Array<T>* createSubArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend Array<T>* createRefArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend void      destroyArray<T>(Array<T> &arr);
        friend Array<T>* createNodeArray<T>(const af::dim4 &dims, TNJ::Node *node);

        friend void *getDevicePtr<T>(const Array<T>& arr);
    };

}
