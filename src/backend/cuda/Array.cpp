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
#include <err_cuda.hpp>
#include <JIT/BufferNode.hpp>
#include <scalar.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <cstddef>

using af::dim4;

namespace cuda
{

    const int MAX_JIT_LEN = 20;
    using JIT::BufferNode;
    using JIT::Node;
    using JIT::Node_ptr;

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        info(getActiveDeviceId(), dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
        node(), offset(0), ready(true), owner(true)
    {}

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data, bool is_device) :
        info(getActiveDeviceId(), dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data((is_device ? (T *)in_data : memAlloc<T>(dims.elements())), memFree<T>),
        data_dims(dims),
        node(), offset(0), ready(true), owner(true)
    {
#if __cplusplus > 199711L
        static_assert(std::is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
        static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
#endif
        if (!is_device) {
            CUDA_CHECK(cudaMemcpy(data.get(), in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice));
        }
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
    Array<T>::Array(Param<T> &tmp) :
        info(getActiveDeviceId(), af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2], tmp.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.ptr, memFree<T>),
        data_dims(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3])),
        node(), offset(0), ready(true), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
        info(-1, dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(), data_dims(dims),
        node(n), offset(0), ready(false), owner(true)
    {
    }


    template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

        this->setId(getActiveDeviceId());
        data = shared_ptr<T>(memAlloc<T>(elements()),
                             memFree<T>);

        Param<T> res;
        res.ptr = data.get();

        for (int  i = 0; i < 4; i++) {
            res.dims[i] = dims()[i];
            res.strides[i] = strides()[i];
        }

        evalNodes(res, this->getNode().get());
        ready = true;

        Node_ptr prev = node;
        prev->resetFlags();
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
    Array<T>::~Array() {}

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {
            bool is_linear = isLinear();
            unsigned bytes = this->getDataDims().elements() * sizeof(T);
            BufferNode<T> *buf_node = new BufferNode<T>(irname<T>(),
                                                        afShortName<T>(), data,
                                                        *this, bytes, is_linear);
            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    template<typename T>
    Array<T> createNodeArray(const dim4 &dims, Node_ptr node)
    {
        Array<T> out =  Array<T>(dims, node);

        unsigned length =0, buf_count = 0, bytes = 0;

        Node *n = node.get();
        n->getInfo(length, buf_count, bytes);
        n->resetFlags();

        if (length > MAX_JIT_LEN ||
            buf_count >= MAX_BUFFERS ||
            bytes >= MAX_BYTES) {
            out.eval();
        }

        return out;
    }

    template<typename T>
    Array<T> createHostDataArray(const dim4 &size, const T * const data)
    {
        return Array<T>(size, data, false);
    }

    template<typename T>
    Array<T> createDeviceDataArray(const dim4 &size, const void *data)
    {
        return Array<T>(size, (const T * const)data, true);
    }

    template<typename T>
    Array<T> createValueArray(const dim4 &size, const T& value)
    {
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T> createEmptyArray(const dim4 &size)
    {
        return Array<T>(size);
    }

    template<typename T>
    Array<T> *initArray()
    {
        return new Array<T>(dim4());
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
    Array<T> createParamArray(Param<T> &tmp)
    {
        return Array<T>(tmp);
    }

    template<typename T>
    void destroyArray(Array<T> *A)
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
        if (!arr.isOwner()) {
            arr = createEmptyArray<T>(arr.dims());
        }

        T *ptr = arr.get();

        CUDA_CHECK(cudaMemcpy(ptr + arr.getOffset(), data,
                              bytes,
                              cudaMemcpyHostToDevice));

        return;
    }

    template<typename T>
    void
    writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes)
    {
        if (!arr.isOwner()) {
            arr = createEmptyArray<T>(arr.dims());
        }

        T *ptr = arr.get();

        CUDA_CHECK(cudaMemcpy(ptr + arr.getOffset(), data,
                              bytes,
                              cudaMemcpyDeviceToDevice));

        return;
    }

#define INSTANTIATE(T)                                                  \
    template       Array<T>  createHostDataArray<T>   (const dim4 &size, const T * const data); \
    template       Array<T>  createDeviceDataArray<T> (const dim4 &size, const void *data); \
    template       Array<T>  createValueArray<T>      (const dim4 &size, const T &value); \
    template       Array<T>  createEmptyArray<T>      (const dim4 &size); \
    template       Array<T>  *initArray<T      >      ();               \
    template       Array<T>  createParamArray<T>      (Param<T> &tmp);  \
    template       Array<T>  createSubArray<T>        (const Array<T> &parent, \
                                                       const std::vector<af_seq> &index, \
                                                       bool copy);      \
    template       void      destroyArray<T>          (Array<T> *A);    \
    template       void      evalArray<T>             (const Array<T> &A); \
    template       Array<T>  createNodeArray<T>       (const dim4 &size, JIT::Node_ptr node); \
    template       Array<T>::~Array        ();                          \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
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

}
