/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <iostream>
#include <TNJ/BufferNode.hpp>
#include <TNJ/ScalarNode.hpp>
#include <memory.hpp>

namespace cpu
{
    using TNJ::BufferNode;
    using TNJ::Node;
    using TNJ::Node_ptr;

    using af::dim4;

    template<typename T>
    Array<T>::Array(dim4 dims):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>),
        node(), ready(true), offset(0), owner(true)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, const T * const in_data):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>),
        node(), ready(true), offset(0), owner(true)
    {
        std::copy(in_data, in_data + dims.elements(), data.get());
    }


    template<typename T>
    Array<T>::Array(af::dim4 dims, TNJ::Node_ptr n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        node(n), ready(false), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim4 &offsets, const dim4 &strides) :
        ArrayInfo(dims, offsets, strides, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()),
        node(), ready(true),
        offset(parent.getOffset() + calcOffset(parent.strides(), offsets)),
        owner(false)
    { }

        template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

        data = std::shared_ptr<T>(memAlloc<T>(elements()), memFree<T>);
        T *ptr = data.get();

        dim4 ostrs = strides();
        dim4 odims = dims();

        for (int w = 0; w < odims[3]; w++) {
            dim_type offw = w * ostrs[3];

            for (int z = 0; z < odims[2]; z++) {
                dim_type offz = z * ostrs[2] + offw;

                for (int y = 0; y < odims[1]; y++) {
                    dim_type offy = y * ostrs[1] + offz;

                    for (int x = 0; x < odims[0]; x++) {
                        dim_type id = x + offy;

                        ptr[id] = *(T *)node->calc(x, y, z, w);
                    }
                }
            }
        }


        ready = true;

        Node_ptr prev = node;
        prev->reset();
        // FIXME: Replace the current node in any JIT possible trees with the new BufferNode
        node.reset();
    }

    template<typename T>
    void Array<T>::eval() const
    {
        if (isReady()) return;
        const_cast<Array<T> *>(this)->eval();
    }

    template<typename T>
    Array<T>::~Array()
    { }

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {

            BufferNode<T> *buf_node = new BufferNode<T>(data,
                                                        dims().get(),
                                                        strides().get(),
                                                        offset);

            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    template<typename T>
    Array<T>
    createHostDataArray(const dim4 &size, const T * const data)
    {
        return Array<T>(size, data);
    }

    template<typename T>
    Array<T>
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        return Array<T>(size, (const T * const) data);
    }

    template<typename T>
    Array<T>
    createValueArray(const dim4 &size, const T& value)
    {
        TNJ::ScalarNode<T> *node = new TNJ::ScalarNode<T>(value);
        return createNodeArray<T>(size, TNJ::Node_ptr(
                                      reinterpret_cast<TNJ::Node *>(node)));
    }

    template<typename T>
    Array<T>
    createEmptyArray(const dim4 &size)
    {
        return Array<T>(size);
    }

    template<typename T>
    Array<T> *initArray() { return new Array<T>(dim4()); }


    template<typename T>
    Array<T>
    createNodeArray(const dim4 &dims, Node_ptr node)
    {
        return Array<T>(dims, node);
    }


    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy)
    {
        dim4 dims   = af::toDims  (index, parent.dims());
        dim4 offset = af::toOffset(index, parent.dims());
        dim4 stride = af::toStride (index, parent.dims());

        Array<T> out = Array<T>(parent, dims, offset, stride);

        if (!copy) return out;

        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {

            out = copyArray(out);
        }

        return out;
    }

    template<typename T>
    void
    destroyArray(Array<T> *A)
    {
        delete A;
    }


    template<typename T>
    void evalArray(const Array<T> &A)
    {
        A.eval();
    }


#define INSTANTIATE(T)                                                  \
    template       Array<T>  createHostDataArray<T>   (const dim4 &size, const T * const data); \
    template       Array<T>  createDeviceDataArray<T> (const dim4 &size, const void *data); \
    template       Array<T>  createValueArray<T>      (const dim4 &size, const T &value); \
    template       Array<T>  createEmptyArray<T>      (const dim4 &size); \
    template       Array<T>  *initArray<T      >      ();               \
    template       Array<T>  createSubArray<T>        (const Array<T> &parent, \
                                                       const std::vector<af_seq> &index, \
                                                       bool copy);      \
    template       void      destroyArray<T>          (Array<T> *A);    \
    template       void      evalArray<T>             (const Array<T> &A); \
    template       Array<T>  createNodeArray<T>       (const dim4 &size, TNJ::Node_ptr node); \
    template       Array<T>::~Array        ();                          \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       TNJ::Node_ptr Array<T>::getNode() const;             \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
}
