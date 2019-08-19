/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

#include <common/half.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/util.hpp>
#include <common/traits.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <jit/BufferNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <scalar.hpp>
#include <af/dim4.hpp>
#include <af/opencl.h>

#include <cstddef>
#include <numeric>

using af::dim4;

using cl::Buffer;

using common::half;
using common::Node;
using common::Node_ptr;
using common::NodeIterator;
using opencl::jit::BufferNode;

using std::accumulate;
using std::is_standard_layout;
using std::make_shared;
using std::vector;

namespace opencl {
template<typename T>
Node_ptr bufferNodePtr() {
    return make_shared<BufferNode>(dtype_traits<T>::getName(),
                                   shortname<T>(true));
}

template<typename T>
Array<T>::Array(dim4 dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data(bufferAlloc(info.elements() * sizeof(T)), bufferFree)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {}

template<typename T>
Array<T>::Array(dim4 dims, Node_ptr n)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data()
    , data_dims(dims)
    , node(n)
    , ready(false)
    , owner(true) {}

template<typename T>
Array<T>::Array(dim4 dims, const T *const in_data)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data(bufferAlloc(info.elements() * sizeof(T)), bufferFree)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
    static_assert(is_standard_layout<Array<T>>::value,
                  "Array<T> must be a standard layout type");
    static_assert(
        offsetof(Array<T>, info) == 0,
        "Array<T>::info must be the first member variable of Array<T>");
    getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0,
                                  sizeof(T) * info.elements(), in_data);
}

template<typename T>
Array<T>::Array(dim4 dims, cl_mem mem, size_t src_offset, bool copy)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data(copy ? bufferAlloc(info.elements() * sizeof(T)) : new Buffer(mem),
           bufferFree)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
    if (copy) {
        clRetainMemObject(mem);
        Buffer src_buf = Buffer((cl_mem)(mem));
        getQueue().enqueueCopyBuffer(src_buf, *data.get(), src_offset, 0,
                                     sizeof(T) * info.elements());
    }
}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &stride)
    : info(parent.getDevId(), dims, offset_, stride,
           (af_dtype)dtype_traits<T>::af_type)
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(false) {}

template<typename T>
Array<T>::Array(Param &tmp, bool owner_)
    : info(getActiveDeviceId(),
           dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2],
                tmp.info.dims[3]),
           0,
           dim4(tmp.info.strides[0], tmp.info.strides[1], tmp.info.strides[2],
                tmp.info.strides[3]),
           (af_dtype)dtype_traits<T>::af_type)
    , data(tmp.data, owner_ ? bufferFree : [](Buffer *) {})
    , data_dims(dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2],
                     tmp.info.dims[3]))
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(owner_) {}

template<typename T>
Array<T>::Array(dim4 dims, dim4 strides, dim_t offset_, const T *const in_data,
                bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           (af_dtype)dtype_traits<T>::af_type)
    , data(is_device ? (new Buffer((cl_mem)in_data))
                     : (bufferAlloc(info.total() * sizeof(T))),
           bufferFree)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
    if (!is_device) {
        getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0,
                                      sizeof(T) * info.total(), in_data);
    }
}

template<typename T>
void Array<T>::eval() {
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
    node  = bufferNodePtr<T>();
}

template<typename T>
void Array<T>::eval() const {
    if (isReady()) return;
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
Buffer *Array<T>::device() {
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->get();
}

template<typename T>
void evalMultiple(vector<Array<T> *> arrays) {
    vector<Param> outputs;
    vector<Array<T> *> output_arrays;
    vector<Node *> nodes;

    for (Array<T> *array : arrays) {
        if (array->isReady()) { continue; }

        const ArrayInfo info = array->info;

        array->ready = true;
        array->setId(getActiveDeviceId());
        array->data =
            Buffer_ptr(bufferAlloc(info.elements() * sizeof(T)), bufferFree);

        // Do not replace this with cast operator
        KParam kInfo = {
            {info.dims()[0], info.dims()[1], info.dims()[2], info.dims()[3]},
            {info.strides()[0], info.strides()[1], info.strides()[2],
             info.strides()[3]},
            0};

        Param res = {array->data.get(), kInfo};

        outputs.push_back(res);
        output_arrays.push_back(array);
        nodes.push_back(array->node.get());
    }
    evalNodes(outputs, nodes);
    for (Array<T> *array : output_arrays) { array->node = bufferNodePtr<T>(); }
}

template<typename T>
Array<T>::~Array() {}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node->isBuffer()) {
        KParam kinfo        = *this;
        BufferNode *bufNode = reinterpret_cast<BufferNode *>(node.get());
        unsigned bytes      = this->getDataDims().elements() * sizeof(T);
        bufNode->setData(kinfo, data, bytes, isLinear());
    }
    return node;
}

template<typename T>
Node_ptr Array<T>::getNode() const {
    if (node->isBuffer()) { return const_cast<Array<T> *>(this)->getNode(); }
    return node;
}

/// This function should be called after a new JIT node is created. It will
/// return true if the newly created node will generate a valid kernel. If
/// false the node will fail to compile or the node and its referenced buffers
/// are consuming too many resources. If false, the node's child nodes should
/// be evaluated before continuing.
///
/// We eval in the following cases:
///
/// 1. Too many bytes are locked up by JIT causing memory
///    pressure. Too many bytes is assumed to be half of all bytes
///    allocated so far.
///
/// 2. The number of parameters we are passing into the kernel exceeds the
///    limitation on the platform. For NVIDIA this is 4096 bytes. The
template<typename T>
kJITHeuristics passesJitHeuristics(Node *root_node) {
    if (!evalFlag()) { return kJITHeuristics::Pass; }
    if (root_node->getHeight() >= (int)getMaxJitSize()) { return kJITHeuristics::TreeHeight; }

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemoryInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    bool isBufferLimit =
        lock_bytes > getMaxBytes() || lock_buffers > getMaxBuffers();
    auto platform = getActivePlatform();

    // The Apple platform can have the nvidia card or the AMD card
    bool isNvidia =
        platform == AFCL_PLATFORM_NVIDIA || platform == AFCL_PLATFORM_APPLE;
    bool isAmd =
        platform == AFCL_PLATFORM_AMD || platform == AFCL_PLATFORM_APPLE;

    // A lightweight check based on the height of the node. This is
    // an inexpensive operation and does not traverse the JIT tree.
    bool isParamLimit = (root_node->getHeight() > 6);
    if (isParamLimit || isBufferLimit) {
        // This is the base parameter size if the kernel had no
        // arguments
        constexpr size_t base_param_size =
            sizeof(T *) + sizeof(KParam) + (3 * sizeof(uint));

        // This is the maximum size of the params that can be allowed by the
        // CUDA platform.
        constexpr size_t max_nvidia_param_size = (4096 - base_param_size);
        constexpr size_t max_amd_param_size    = (3520 - base_param_size);

        size_t max_param_size = 0;
        if (isNvidia) {
            max_param_size = max_nvidia_param_size;
        } else if (isAmd) {
            max_param_size = max_amd_param_size;
        } else {
            max_param_size = 8192;
        }

        struct tree_info {
            size_t total_buffer_size;
            size_t num_buffers;
            size_t param_scalar_size;
        };
        NodeIterator<> it(root_node);
        tree_info info =
            accumulate(it, NodeIterator<>(), tree_info{0, 0, 0},
                       [](tree_info &prev, Node &n) {
                           if (n.isBuffer()) {
                               auto &buf_node = static_cast<BufferNode &>(n);
                               // getBytes returns the size of the data Array.
                               // Sub arrays will be represented by their parent
                               // size.
                               prev.total_buffer_size += buf_node.getBytes();
                               prev.num_buffers++;
                           } else {
                               prev.param_scalar_size += n.getParamBytes();
                           }
                           return prev;
                       });
        isBufferLimit = 2 * info.total_buffer_size > lock_bytes;

        size_t param_size = (info.num_buffers * (sizeof(KParam) + sizeof(T *)) +
                             info.param_scalar_size);

        isParamLimit = param_size >= max_param_size;

        if (isParamLimit) {
            return kJITHeuristics::KernelParameterSize;
        }
        if (isBufferLimit) {
            return kJITHeuristics::MemoryPressure;
        }
    }
    return kJITHeuristics::Pass;
}

template<typename T>
Array<T> createNodeArray(const dim4 &dims, Node_ptr node) {
    verifyTypeSupport<T>();
    Array<T> out = Array<T>(dims, node);
    return out;
}

template<typename T>
Array<T> createSubArray(const Array<T> &parent, const vector<af_seq> &index,
                        bool copy) {
    parent.eval();

    dim4 dDims          = parent.getDataDims();
    dim4 dStrides       = calcStrides(dDims);
    dim4 parent_strides = parent.strides();

    if (dStrides != parent_strides) {
        const Array<T> parentCopy = copyArray(parent);
        return createSubArray(parentCopy, index, copy);
    }

    dim4 pDims = parent.dims();

    dim4 dims    = toDims(index, pDims);
    dim4 strides = toStride(index, dDims);

    // Find total offsets after indexing
    dim4 offsets = toOffset(index, pDims);
    dim_t offset = parent.getOffset();
    for (int i = 0; i < 4; i++) offset += offsets[i] * parent_strides[i];

    Array<T> out = Array<T>(parent, dims, offset, strides);

    if (!copy) return out;

    if (strides[0] != 1 || strides[1] < 0 || strides[2] < 0 || strides[3] < 0) {
        out = copyArray(out);
    }

    return out;
}

template<typename T>
Array<T> createHostDataArray(const dim4 &size, const T *const data) {
    verifyTypeSupport<T>();
    return Array<T>(size, data);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &size, void *data) {
    verifyTypeSupport<T>();

    bool copy_device = false;
    return Array<T>(size, static_cast<cl_mem>(data), 0, copy_device);
}

template<typename T>
Array<T> createValueArray(const dim4 &size, const T &value) {
    verifyTypeSupport<T>();
    return createScalarNode<T>(size, value);
}

template<typename T>
Array<T> createEmptyArray(const dim4 &size) {
    verifyTypeSupport<T>();
    return Array<T>(size);
}

template<typename T>
Array<T> createParamArray(Param &tmp, bool owner) {
    verifyTypeSupport<T>();
    return Array<T>(tmp, owner);
}

template<typename T>
void destroyArray(Array<T> *A) {
    delete A;
}

template<typename T>
void writeHostDataArray(Array<T> &arr, const T *const data,
                        const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }

    getQueue().enqueueWriteBuffer(*arr.get(), CL_TRUE, arr.getOffset(), bytes,
                                  data);

    return;
}

template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }

    Buffer &buf = *arr.get();

    clRetainMemObject((cl_mem)(data));
    Buffer data_buf = Buffer((cl_mem)(data));

    getQueue().enqueueCopyBuffer(data_buf, buf, 0, (size_t)arr.getOffset(),
                                 bytes);

    return;
}

template<typename T>
void Array<T>::setDataDims(const dim4 &new_dims) {
    modDims(new_dims);
    data_dims = new_dims;
    if (node->isBuffer()) { node = bufferNodePtr<T>(); }
}

#define INSTANTIATE(T)                                                        \
    template Array<T> createHostDataArray<T>(const dim4 &dims,                \
                                             const T *const data);            \
    template Array<T> createDeviceDataArray<T>(const dim4 &dims, void *data); \
    template Array<T> createValueArray<T>(const dim4 &dims, const T &value);  \
    template Array<T> createEmptyArray<T>(const dim4 &dims);                  \
    template Array<T> createParamArray<T>(Param & tmp, bool owner);           \
    template Array<T> createSubArray<T>(                                      \
        const Array<T> &parent, const vector<af_seq> &index, bool copy);      \
    template void destroyArray<T>(Array<T> * A);                              \
    template Array<T> createNodeArray<T>(const dim4 &dims, Node_ptr node);    \
    template Array<T>::Array(dim4 dims, dim4 strides, dim_t offset,           \
                             const T *const in_data, bool is_device);         \
    template Array<T>::Array(dim4 dims, cl_mem mem, size_t src_offset,        \
                             bool copy);                                      \
    template Array<T>::~Array();                                              \
    template Node_ptr Array<T>::getNode() const;                              \
    template void Array<T>::eval();                                           \
    template void Array<T>::eval() const;                                     \
    template Buffer *Array<T>::device();                                      \
    template void writeHostDataArray<T>(Array<T> & arr, const T *const data,  \
                                        const size_t bytes);                  \
    template void writeDeviceDataArray<T>(                                    \
        Array<T> & arr, const void *const data, const size_t bytes);          \
    template void evalMultiple<T>(vector<Array<T> *> arrays);                 \
    template kJITHeuristics passesJitHeuristics<T>(Node * node);              \
    template void Array<T>::setDataDims(const dim4 &new_dims);

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
INSTANTIATE(half)

}  // namespace opencl
