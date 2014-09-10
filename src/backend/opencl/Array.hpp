#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <kernel/set.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include "traits.hpp"
#include <backend.hpp>
#include <types.hpp>
#include <traits.hpp>

namespace opencl
{
    using kernel::set;
    using af::dim4;

    template<typename T> class Array;

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createValueArray(const af::dim4 &size, const T& value);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createDataArray(const af::dim4 &size, const T * const data);

    // Create an Array object and do not assign any values to it
    template<typename T>
    Array<T>*
    createEmptyArray(const af::dim4 &size);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    void
    destroyArray(Array<T> &A);

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const af::dim4 &dims, const af::dim4 &offset, const af::dim4 &stride);


    template<typename T>
    class Array : public ArrayInfo
    {
        cl::Buffer  data;
        const Array*      parent;

        Array(af::dim4 dims);
        explicit Array(af::dim4 dims, T val);
        explicit Array(af::dim4 dims, const T * const in_data);
        Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    public:

        ~Array();

        bool isOwner() const { return parent == nullptr; }


        //FIXME: This should do a copy if it is not owner. You do not want to overwrite parents data
        cl::Buffer& get()
        {
            if (isOwner()) return data;
            return (cl::Buffer &)parent->data;
        }

        const   cl::Buffer& get() const
        {
            if (isOwner()) return data;
            return parent->data;
        }

        const dim_type getOffset() const
        {
            return isOwner() ? 0 : calcOffset(parent->strides(), this->offsets());
        }

        friend Array<T>* createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T>* createDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T>* createEmptyArray<T>(const af::dim4 &size);
        friend Array<T>* createSubArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend void      destroyArray<T>(Array<T> &arr);
    };
}
