/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <JIT/BufferNode.hpp>
#include <af/dim4.hpp>
#include <af/opencl.h>
#include <common/NodeIterator.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <scalar.hpp>

#include <cstddef>
#include <numeric>

using af::dim4;
using common::NodeIterator;
using opencl::JIT::BufferNode;
using opencl::JIT::Node;
using opencl::JIT::Node_ptr;
using std::accumulate;

namespace opencl
{
    template<typename T>
    Node_ptr bufferNodePtr()
    {
        return std::make_shared<BufferNode>(dtype_traits<T>::getName(), shortname<T>(true));
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(info.elements() * sizeof(T)), bufferFree),
        data_dims(dims),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        data_dims(dims),
        node(n), ready(false), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(info.elements()*sizeof(T)), bufferFree),
        data_dims(dims),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {
        static_assert(std::is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
        static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
        getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0, sizeof(T)*info.elements(), in_data);
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, cl_mem mem, size_t src_offset, bool copy) :
        info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(copy ? bufferAlloc(info.elements() * sizeof(T)) : new cl::Buffer(mem), bufferFree),
        data_dims(dims),
        node(bufferNodePtr<T>()), ready(true), owner(true)
    {
        if (copy) {
            clRetainMemObject(mem);
            cl::Buffer src_buf = cl::Buffer((cl_mem)(mem));
            getQueue().enqueueCopyBuffer(src_buf, *data.get(),
                                         src_offset, 0,
                                         sizeof(T) * info.elements());
        }
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim_t &offset_, const dim4 &stride) :
        info(parent.getDevId(), dims, offset_, stride, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()),
        data_dims(parent.getDataDims()),
        node(bufferNodePtr<T>()),
        ready(true),
        owner(false)
    {
    }


    template<typename T>
    Array<T>::Array(Param &tmp, bool owner_) :
        info(getActiveDeviceId(),
             af::dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2], tmp.info.dims[3]),
             0,
             af::dim4(tmp.info.strides[0], tmp.info.strides[1],
                      tmp.info.strides[2], tmp.info.strides[3]),
             (af_dtype)dtype_traits<T>::af_type),
        data(tmp.data, owner_ ? bufferFree : [] (cl::Buffer* ptr) {}),
        data_dims(af::dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2], tmp.info.dims[3])),
        node(bufferNodePtr<T>()), ready(true), owner(owner_)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset_,
                    const T * const in_data, bool is_device) :
        info(getActiveDeviceId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
        data(is_device ?
             (new cl::Buffer((cl_mem)in_data)) :
             (bufferAlloc(info.total() * sizeof(T))), bufferFree),
        data_dims(dims),
        node(bufferNodePtr<T>()),
        ready(true),
        owner(true)
    {
        if (!is_device) {
            getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0, sizeof(T) * info.total(), in_data);
        }
    }

    template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

        this->setId(getActiveDeviceId());
        data = Buffer_ptr(bufferAlloc(elements() * sizeof(T)), bufferFree);

        // Do not replace this with cast operator
        KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                       {strides()[0], strides()[1], strides()[2], strides()[3]},
                       0};

        Param res = {data.get(), info};

        evalNodes(res, node.get());
        ready = true;
        node = bufferNodePtr<T>();
    }

    template<typename T>
    void Array<T>::eval() const
    {
        if (isReady()) return;
        const_cast<Array<T> *>(this)->eval();
    }

    template<typename T>
    cl::Buffer* Array<T>::device()
    {
        if (!isOwner() || getOffset() || data.use_count() > 1) {
            *this = copyArray<T>(*this);
        }
        return this->get();
    }

    template<typename T>
    void evalMultiple(std::vector<Array<T>*> arrays)
    {
        std::vector<Param> outputs;
        std::vector<Node *> nodes;

        for (auto array : arrays) {
            if (array->isReady()) {
                continue;
            }

            const ArrayInfo info = array->info;

            array->setId(getActiveDeviceId());
            array->data = Buffer_ptr(bufferAlloc(info.elements() * sizeof(T)), bufferFree);

            // Do not replace this with cast operator
            KParam kInfo = {{info.dims()[0], info.dims()[1], info.dims()[2], info.dims()[3]},
                            {info.strides()[0], info.strides()[1],
                             info.strides()[2], info.strides()[3]},
                            0};

            Param res = {array->data.get(), kInfo};
            outputs.push_back(res);
            nodes.push_back(array->node.get());
        }
        evalNodes(outputs, nodes);
        for (auto array : arrays) {
            if (array->isReady()) continue;
            array->ready = true;
            array->node = bufferNodePtr<T>();
        }
    }

    template<typename T>
    Array<T>::~Array()
    {
    }

    template<typename T>
    Node_ptr Array<T>::getNode()
    {
        if (node->isBuffer()) {
            KParam kinfo = *this;
            BufferNode *bufNode = reinterpret_cast<BufferNode *>(node.get());
            unsigned bytes = this->getDataDims().elements() * sizeof(T);
            bufNode->setData(kinfo, data, bytes, isLinear());
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

    using af::dim4;

    template<typename T>
    Array<T> createNodeArray(const dim4 &dims, Node_ptr node)
    {
        verifyDoubleSupport<T>();
        Array<T> out =  Array<T>(dims, node);

        if (evalFlag()) {

            if (node->getHeight() >= (int)getMaxJitSize()) {
                out.eval();
            } else {

                size_t alloc_bytes, alloc_buffers;
                size_t lock_bytes, lock_buffers;

                deviceMemoryInfo(&alloc_bytes, &alloc_buffers,
                                 &lock_bytes, &lock_buffers);

                bool isBufferLimit =
                    lock_bytes > getMaxBytes() ||
                    lock_buffers > getMaxBuffers();


                bool isNvidia = getActivePlatform() == AFCL_PLATFORM_NVIDIA;
                // We eval in the following cases.
                // 1. Too many bytes are locked up by JIT causing memory pressure.
                // Too many bytes is assumed to be half of all bytes allocated so far.
                // 2. Too many buffers in a nonlinear kernel cause param space overflow.
                // Too many buffers comes out to be about 48 (49 including output).
                // Too many buffers can occur in a tree of size 24 in the worst case scenario.
                // This error only happens on nvidia devices.
                // TODO: Find better solution than the following emperical solution.
                bool isParamLimit = (isNvidia && node->getHeight() > 24);
                if (isParamLimit || isBufferLimit) {
                    // This is the maximum non-linear buffers that are allowed in
                    // the parameter list
                    constexpr int max_nonlinear_buffer_count = 48;

                    Node *n = node.get();

                    struct  tree_info {
                        size_t buffer_size;
                        int num_buffers;
                        bool is_linear;
                    };
                    NodeIterator it(n);
                    NodeIterator end_node;
                    dim4 outdim = out.dims();
                    tree_info info = accumulate(it, end_node,
                                                tree_info{0, 0, true},
                                                [=](tree_info& prev, Node& n) {
                                                    if(n.isBuffer()) {
                                                        auto& buf_node = static_cast<BufferNode&>(n);
                                                        prev.buffer_size += buf_node.getBytes();
                                                        prev.num_buffers++;
                                                        prev.is_linear &= buf_node.isLinear((dim_t*)outdim.get());
                                                    }
                                                    // getBytes returns the size of the data Array. Sub arrays will
                                                    // be represented by their parent size.
                                                    return prev;
                                                });
                    isBufferLimit = 2 * info.buffer_size > lock_bytes;
                    isParamLimit = isNvidia &&
                                   !info.is_linear &&
                                   info.num_buffers >= max_nonlinear_buffer_count;

                    if (isBufferLimit || isParamLimit) {
                        out.eval();
                    }
                }
            }
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
        dim4 dStrides = calcStrides(dDims);
        dim4 parent_strides = parent.strides();

        if (dStrides != parent_strides) {
            const Array<T> parentCopy = copyArray(parent);
            return createSubArray(parentCopy, index, copy);
        }

        dim4 pDims = parent.dims();

        dim4 dims    = toDims  (index, pDims);
        dim4 strides = toStride (index, dDims);

        // Find total offsets after indexing
        dim4 offsets = toOffset(index, pDims);
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
    Array<T>
    createHostDataArray(const dim4 &size, const T * const data)
    {
        verifyDoubleSupport<T>();
        return Array<T>(size, data);
    }

    template<typename T>
    Array<T>
    createDeviceDataArray(const dim4 &size, const void *data, bool copy)
    {
        verifyDoubleSupport<T>();

        return Array<T>(size, (cl_mem)(data), 0, copy);
    }

    template<typename T>
    Array<T>
    createValueArray(const dim4 &size, const T& value)
    {
        verifyDoubleSupport<T>();
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T>
    createEmptyArray(const dim4 &size)
    {
        verifyDoubleSupport<T>();
        return Array<T>(size);
    }

    template<typename T>
    Array<T> *initArray()
    {
        return new Array<T>(dim4());
    }

    template<typename T>
    Array<T>
    createParamArray(Param &tmp, bool owner)
    {
        verifyDoubleSupport<T>();
        return Array<T>(tmp, owner);
    }

    template<typename T>
    void
    destroyArray(Array<T> *A)
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

        getQueue().enqueueWriteBuffer(*arr.get(), CL_TRUE,
                                      arr.getOffset(),
                                      bytes,
                                      data);

        return;
    }

    template<typename T>
    void
    writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes)
    {
        if (!arr.isOwner()) {
            arr = copyArray<T>(arr);
        }

        cl::Buffer& buf = *arr.get();

        clRetainMemObject((cl_mem)(data));
        cl::Buffer data_buf = cl::Buffer((cl_mem)(data));

        getQueue().enqueueCopyBuffer(data_buf, buf,
                                     0, (size_t)arr.getOffset(),
                                     bytes);

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
    template       Array<T>  createDeviceDataArray<T> (const dim4 &size, const void *data, bool copy); \
    template       Array<T>  createValueArray<T>      (const dim4 &size, const T &value); \
    template       Array<T>  createEmptyArray<T>      (const dim4 &size); \
    template       Array<T>  *initArray<T      >      ();               \
    template       Array<T>  createParamArray<T>      (Param &tmp, bool owner);    \
    template       Array<T>  createSubArray<T>        (const Array<T> &parent, \
                                                       const std::vector<af_seq> &index, \
                                                       bool copy);      \
    template       void      destroyArray<T>          (Array<T> *A);    \
    template       Array<T>  createNodeArray<T>       (const dim4 &size, JIT::Node_ptr node); \
    template       Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset, \
                                   const T * const in_data,             \
                                   bool is_device);                     \
    template       Array<T>::Array(af::dim4 dims, cl_mem mem, size_t src_offset, bool copy); \
    template       Array<T>::~Array        ();                          \
    template       Node_ptr Array<T>::getNode() const;                  \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       cl::Buffer* Array<T>::device();                      \
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
