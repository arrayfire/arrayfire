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

namespace cuda
{

    using af::dim4;
    template<typename T> class Array;

    template<typename T>
    void evalNodes(Param<T> &out, JIT::Node *node);

    // Creates a new Array object on the heap and returns a reference to it.
    template<typename T>
    Array<T>*
    createNodeArray(const af::dim4 &size, JIT::Node *node);

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

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor=1.0);

    template<typename T>
    T* cudaMallocWrapper(const size_t &elements);

    template<typename T>
    class Array : public ArrayInfo
    {
        T*      data;
        const Array*  parent;

        JIT::Node *node;
        bool ready;

        Array(af::dim4 dims);
        explicit Array(af::dim4 dims, T val);
        explicit Array(af::dim4 dims, const T * const in_data);
        Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
        Array(Param<T> &tmp);
        Array(af::dim4 dims, JIT::Node *n);
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
            if (isOwner()) return data;
            return parent->data + (withOffset ? calcOffset(parent->strides(), this->offsets()) : 0);
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

        JIT::Node *getNode() const;

        friend Array<T>* createValueArray<T>(const af::dim4 &size, const T& value);
        friend Array<T>* createDataArray<T>(const af::dim4 &size, const T * const data);
        friend Array<T>* createEmptyArray<T>(const af::dim4 &size);
        friend Array<T>* createParamArray<T>(Param<T> &tmp);
        friend Array<T>* createNodeArray<T>(const af::dim4 &dims, JIT::Node *node);
        friend Array<T>* createSubArray<T>(const Array<T>& parent,
                                           const dim4 &dims, const dim4 &offset, const dim4 &stride);
        friend void      destroyArray<T>(Array<T> &arr);
    };

}
