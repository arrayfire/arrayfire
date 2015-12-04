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
#include <TNJ/BufferNode.hpp>
#include <TNJ/ScalarNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <cstring>
#include <cstddef>

namespace cpu
{
    const int MAX_TNJ_LEN = 20;
    using TNJ::BufferNode;
    using TNJ::Node;
    using TNJ::Node_ptr;

    using af::dim4;

    template<typename T>
    Array<T>::Array(dim4 dims):
        info(getActiveDeviceId(), dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
        node(), offset(0), ready(true), owner(true)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, const T * const in_data, bool is_device):
        info(getActiveDeviceId(), dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(is_device ? (T*)in_data : memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
        node(), offset(0), ready(true), owner(true)
    {
        static_assert(std::is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
        static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
        if (!is_device) {
            std::copy(in_data, in_data + dims.elements(), data.get());
        }
    }


    template<typename T>
    Array<T>::Array(af::dim4 dims, TNJ::Node_ptr n) :
        info(getActiveDeviceId(), dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(), data_dims(dims),
        node(n), offset(0), ready(false), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim4 &offsets, const dim4 &strides) :
        info(parent.getDevId(), dims, offsets, strides, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()), data_dims(parent.getDataDims()),
        node(),
        offset(parent.getOffset() + calcOffset(parent.strides(), offsets)),
        ready(true), owner(false)
    { }

    template<typename T>
    std::shared_ptr<T> evalNodes(const int &num,
                                 const dim4 &odims,
                                 const dim4 &ostrs,
                                 TNJ::Node_ptr &node)
    {

        std::shared_ptr<T> data(memAlloc<T>(num), memFree<T>);
        T *ptr = data.get();

        bool is_linear = node->isLinear(odims.get());

        if (is_linear) {
            for (int i = 0; i < num; i++) {
                ptr[i] = *(T *)node->calc(i);
            }
        } else {
            for (int w = 0; w < (int)odims[3]; w++) {
                dim_t offw = w * ostrs[3];

                for (int z = 0; z < (int)odims[2]; z++) {
                    dim_t offz = z * ostrs[2] + offw;

                    for (int y = 0; y < (int)odims[1]; y++) {
                        dim_t offy = y * ostrs[1] + offz;

                        for (int x = 0; x < (int)odims[0]; x++) {
                            dim_t id = x + offy;

                            ptr[id] = *(T *)node->calc(x, y, z, w);
                        }
                    }
                }
            }
        }

        return data;
    }

    template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

        this->setId(getActiveDeviceId());

        data = evalNodes<T>(elements(), dims(), strides(), node);

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
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {

            unsigned bytes = this->getDataDims().elements() * sizeof(T);

            BufferNode<T> *buf_node = new BufferNode<T>(data,
                                                        bytes,
                                                        offset,
                                                        dims().get(),
                                                        strides().get(),
                                                        isLinear());

            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    template<typename T>
    Array<T>
    createHostDataArray(const dim4 &size, const T * const data)
    {
        return Array<T>(size, data, false);
    }

    template<typename T>
    Array<T>
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        return Array<T>(size, (const T * const) data, true);
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
    Array<T> *initArray() { return new Array<T>(dim4(0, 0, 0, 0)); }


    template<typename T>
    Array<T>
    createNodeArray(const dim4 &dims, Node_ptr node)
    {
        Array<T> out =  Array<T>(dims, node);

        unsigned length =0, buf_count = 0, bytes = 0;

        Node *n = node.get();
        n->getInfo(length, buf_count, bytes);
        n->reset();

        if (length > MAX_TNJ_LEN ||
            buf_count >= MAX_BUFFERS ||
            bytes >= MAX_BYTES) {
            out.eval();
        }

        return out;
    }


    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy)
    {
        parent.eval();

        dim4 dDims = parent.getDataDims();
        dim4 pDims = parent.dims();

        dim4 dims   = toDims  (index, pDims);
        dim4 offset = toOffset(index, dDims);
        dim4 stride = toStride (index, dDims);

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

    template<typename T>
    void
    writeHostDataArray(Array<T> &arr, const T * const data, const size_t bytes)
    {
        if(!arr.isOwner()) {
            arr = createEmptyArray<T>(arr.dims());
        }
        memcpy(arr.get() + arr.getOffset(), data, bytes);
    }

    template<typename T>
    void
    writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes)
    {
        if(!arr.isOwner()) {
            arr = createEmptyArray<T>(arr.dims());
        }
        memcpy(arr.get() + arr.getOffset(), (const T * const)data, bytes);
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
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       TNJ::Node_ptr Array<T>::getNode() const;             \
    template       void      writeHostDataArray<T>    (Array<T> &arr, const T * const data, const size_t bytes); \
    template       void      writeDeviceDataArray<T>  (Array<T> &arr, const void * const data, const size_t bytes); \

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
    INSTANTIATE(short)
    INSTANTIATE(ushort)
}
