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
#include <memory>

using af::dim4;
using std::shared_ptr;

namespace cuda
{

    const int MAX_JIT_LEN = 20;
    using JIT::BufferNode;
    using JIT::Node;
    using JIT::Node_ptr;

    template<typename T>
    Node_ptr bufferNodePtr()
    {
        return Node_ptr(new BufferNode<T>(getFullName<T>(),
                                          shortname<T>(true)));
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {}

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data, bool is_device, bool copy_device) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(((is_device & !copy_device) ? (T *)in_data : memAlloc<T>(dims.elements())), memFree<T>),
        data_dims(dims),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {
#if __cplusplus > 199711L
        static_assert(std::is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
        static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
#endif
        if (!is_device) {
            CUDA_CHECK(cudaMemcpyAsync(data.get(), in_data, dims.elements() * sizeof(T),
                                       cudaMemcpyHostToDevice, cuda::getActiveStream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        } else if (copy_device) {
            CUDA_CHECK(cudaMemcpyAsync(data.get(), in_data, dims.elements() * sizeof(T),
                                       cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        }
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim_t &offset_, const dim4 &strides) :
        info(parent.getDevId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()), data_dims(parent.getDataDims()),
        node(bufferNodePtr<T>()),
        ready(true), owner(false)
    { }

    template<typename T>
    Array<T>::Array(Param<T> &tmp) :
        info(getActiveDeviceId(),
             af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]),
             0,
             af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2], tmp.strides[3]),
             (af_dtype)dtype_traits<T>::af_type),
        data(tmp.ptr, memFree<T>),
        data_dims(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3])),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(), data_dims(dims),
        node(n), ready(false), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset_,
                    const T * const in_data, bool is_device) :
        info(getActiveDeviceId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
        data(is_device ? (T*)in_data : memAlloc<T>(info.total()), memFree<T>),
        data_dims(dims),
        node(bufferNodePtr<T>()),
        ready(true),
        owner(true)
    {
        if (!is_device) {
            cudaStream_t stream = getActiveStream();
            CUDA_CHECK(cudaMemcpyAsync(data.get(), in_data, info.total() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
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
        // FIXME: Replace the current node in any JIT possible trees with the new BufferNode
        node = bufferNodePtr<T>();
    }

    template<typename T>
    T* Array<T>::device()
    {
        if (!isOwner() || getOffset() || data.use_count() > 1) {
            *this = copyArray<T>(*this);
        }
        return this->get();
    }

    template<typename T>
    void Array<T>::eval() const
    {
        if (isReady()) return;
        const_cast<Array<T> *>(this)->eval();
    }

    template<typename T>
    void evalMultiple(std::vector<Array<T>*> arrays)
    {
        std::vector<Param<T> > outputs;
        std::vector<JIT::Node *> nodes;

        for (int i = 0; i < (int)arrays.size(); i++) {
            Array<T> *array = arrays[i];

            if (array->isReady()) {
                continue;
            }

            array->setId(getActiveDeviceId());
            array->data = shared_ptr<T>(memAlloc<T>(array->elements()),
                                        memFree<T>);

            Param<T> res;
            res.ptr = array->data.get();

            for (int  i = 0; i < 4; i++) {
                res.dims[i] = array->dims()[i];
                res.strides[i] = array->strides()[i];
            }

            outputs.push_back(res);
            nodes.push_back(array->node.get());
        }

        evalNodes(outputs, nodes);

        for (int i = 0; i < (int)arrays.size(); i++) {
            Array<T> *array = arrays[i];

            if (array->isReady()) continue;
            array->ready = true;
            // FIXME: Replace the current node in any JIT possible trees with the new BufferNode
            array->node = bufferNodePtr<T>();
        }
        return;
    }

    template<typename T>
    Array<T>::~Array() {}

    template<typename T>
    Node_ptr Array<T>::getNode()
    {
        if (node->isBuffer()) {
            unsigned bytes = this->getDataDims().elements() * sizeof(T);
            BufferNode<T> *bufNode = reinterpret_cast<BufferNode<T> *>(node.get());
            Param<T> param = *this;
            bufNode->setData(param, data, bytes, isLinear());
        }
        return node;
    }

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (node->isBuffer()) {
            return const_cast<Array<T> *>(this)->getNode();
        }
        return node;
    }

    template<typename T>
    Array<T> createNodeArray(const dim4 &dims, Node_ptr node)
    {
        Array<T> out =  Array<T>(dims, node);

        if (evalFlag()) {

            if (node->getHeight() >= (int)getMaxJitSize()) {
                out.eval();
            } else {
                size_t alloc_bytes, alloc_buffers;
                size_t lock_bytes, lock_buffers;

                deviceMemoryInfo(&alloc_bytes, &alloc_buffers,
                                 &lock_bytes, &lock_buffers);

                // Check if approaching the memory limit
                if (lock_bytes > getMaxBytes() ||
                    lock_buffers > getMaxBuffers()) {

                    unsigned length =0, buf_count = 0, bytes = 0;
                    Node *n = node.get();
                    JIT::Node_map_t nodes_map;
                    n->getNodesMap(nodes_map);

                    for(auto &entry : nodes_map) {
                        Node *node = entry.first;
                        node->getInfo(length, buf_count, bytes);
                    }

                    if (2 * bytes > lock_bytes) {
                        out.eval();
                    }
                }
            }
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

        dim4 dims    = toDims  (index, pDims);
        dim4 strides = toStride (index, dDims);

        // Find total offsets after indexing
        dim4 offsets = toOffset(index, pDims);
        dim4 parent_strides = parent.strides();
        dim_t offset = parent.getOffset();
        for (int i = 0; i < 4; i++) offset += offsets[i] * parent_strides[i];

        Array<T> out = Array<T>(parent, dims, offset, strides);

        if (!copy) return out;

        if (strides[0] != 1 ||
            strides[1] <  0 ||
            strides[2] <  0 ||
            strides[3] <  0) {

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
    void
    writeHostDataArray(Array<T> &arr, const T * const data, const size_t bytes)
    {
        if (!arr.isOwner()) {
            arr = copyArray<T>(arr);
        }

        T *ptr = arr.get();

        CUDA_CHECK(cudaMemcpyAsync(ptr, data, bytes, cudaMemcpyHostToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

        return;
    }

    template<typename T>
    void
    writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes)
    {
        if (!arr.isOwner()) {
            arr = copyArray<T>(arr);
        }

        T *ptr = arr.get();

        CUDA_CHECK(cudaMemcpyAsync(ptr, data, bytes, cudaMemcpyDeviceToDevice, cuda::getActiveStream()));

        return;
    }

    template<typename T>
    void
    Array<T>::setDataDims(const dim4 &new_dims)
    {
        modDims(new_dims);
        data_dims = new_dims;
        if (node->isBuffer()) {
            node = bufferNodePtr<T>();
        }
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
    template       Array<T>  createNodeArray<T>       (const dim4 &size, JIT::Node_ptr node); \
    template       Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset, \
                                   const T * const in_data,             \
                                   bool is_device);                     \
    template       Array<T>::Array(af::dim4 dims, const T * const in_data, \
                                   bool is_device, bool copy_device);   \
    template       Array<T>::~Array        ();                          \
    template       Node_ptr Array<T>::getNode() const;                  \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       T*   Array<T>::device();                             \
    template       void      writeHostDataArray<T>    (Array<T> &arr, const T * const data, \
                                                       const size_t bytes); \
    template       void      writeDeviceDataArray<T>  (Array<T> &arr, const void * const data, \
                                                       const size_t bytes); \
    template       void      evalMultiple<T>     (std::vector<Array<T>*> arrays); \
    template       void Array<T>::setDataDims(const dim4 &new_dims);    \

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
