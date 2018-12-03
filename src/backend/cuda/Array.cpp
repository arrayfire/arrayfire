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
#include <copy.hpp>
#include <err_cuda.hpp>
#include <jit/BufferNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <scalar.hpp>
#include <af/dim4.hpp>

#include <cstddef>
#include <memory>
#include <numeric>

using af::dim4;
using common::half;
using common::Node;
using common::Node_ptr;
using common::NodeIterator;
using cuda::jit::BufferNode;

using std::accumulate;
using std::shared_ptr;
using std::vector;

namespace cuda {
template<typename T>
Node_ptr bufferNodePtr() {
    return Node_ptr(new BufferNode<T>(getFullName<T>(), shortname<T>(true)));
}

template<typename T>
Array<T>::Array(af::dim4 dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data((dims.elements() ? memAlloc<T>(dims.elements()).release() : nullptr),
           memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {}

template<typename T>
Array<T>::Array(af::dim4 dims, const T *const in_data, bool is_device,
                bool copy_device)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data(
          ((is_device & !copy_device) ? const_cast<T *>(in_data)
                                      : memAlloc<T>(dims.elements()).release()),
          memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
#if __cplusplus > 199711L
    static_assert(std::is_standard_layout<Array<T>>::value,
                  "Array<T> must be a standard layout type");
    static_assert(
        offsetof(Array<T>, info) == 0,
        "Array<T>::info must be the first member variable of Array<T>");
#endif
    if (!is_device) {
        CUDA_CHECK(
            cudaMemcpyAsync(data.get(), in_data, dims.elements() * sizeof(T),
                            cudaMemcpyHostToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
    } else if (copy_device) {
        CUDA_CHECK(
            cudaMemcpyAsync(data.get(), in_data, dims.elements() * sizeof(T),
                            cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
    }
}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &strides)
    : info(parent.getDevId(), dims, offset_, strides,
           (af_dtype)dtype_traits<T>::af_type)
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(false) {}

template<typename T>
Array<T>::Array(Param<T> &tmp, bool owner_)
    : info(getActiveDeviceId(),
           af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]), 0,
           af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2],
                    tmp.strides[3]),
           (af_dtype)dtype_traits<T>::af_type)
    , data(tmp.ptr, owner_ ? std::function<void(T *)>(memFree<T>)
                           : std::function<void(T *)>([](T *) {}))
    , data_dims(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]))
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(owner_) {}

template<typename T>
Array<T>::Array(af::dim4 dims, common::Node_ptr n)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           (af_dtype)dtype_traits<T>::af_type)
    , data()
    , data_dims(dims)
    , node(n)
    , ready(false)
    , owner(true) {}

template<typename T>
Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset_,
                const T *const in_data, bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           (af_dtype)dtype_traits<T>::af_type)
    , data(is_device ? (T *)in_data : memAlloc<T>(info.total()).release(),
           memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
    if (!is_device) {
        cudaStream_t stream = getActiveStream();
        CUDA_CHECK(cudaMemcpyAsync(data.get(), in_data,
                                   info.total() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

template<typename T>
void Array<T>::eval() {
    if (isReady()) return;

    this->setId(getActiveDeviceId());
    this->data = shared_ptr<T>(memAlloc<T>(elements()).release(), memFree<T>);

    ready = true;
    evalNodes<T>(*this, this->getNode().get());
    // FIXME: Replace the current node in any JIT possible trees with the new
    // BufferNode
    node = bufferNodePtr<T>();
}

template<typename T>
T *Array<T>::device() {
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->get();
}

template<typename T>
void Array<T>::eval() const {
    if (isReady()) return;
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
void evalMultiple(std::vector<Array<T> *> arrays) {
    vector<Param<T>> outputs;
    vector<Array<T> *> output_arrays;
    vector<Node *> nodes;

    for (Array<T> *array : arrays) {
        if (array->isReady()) { continue; }

        array->ready = true;
        array->setId(getActiveDeviceId());
        array->data =
            shared_ptr<T>(memAlloc<T>(array->elements()).release(), memFree<T>);

        outputs.push_back(*array);
        output_arrays.push_back(array);
        nodes.push_back(array->node.get());
    }

    evalNodes(outputs, nodes);

    for (Array<T> *array : output_arrays) array->node = bufferNodePtr<T>();

    return;
}

template<typename T>
Array<T>::~Array() {}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node->isBuffer()) {
        unsigned bytes         = this->getDataDims().elements() * sizeof(T);
        BufferNode<T> *bufNode = reinterpret_cast<BufferNode<T> *>(node.get());
        Param<T> param         = *this;
        bufNode->setData(param, data, bytes, isLinear());
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
bool passesJitHeuristics(Node *root_node) {
    if (!evalFlag()) return true;
    if (root_node->getHeight() >= (int)getMaxJitSize()) { return false; }

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemoryInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    bool isBufferLimit =
        lock_bytes > getMaxBytes() || lock_buffers > getMaxBuffers();

    // A lightweight check based on the height of the node. This is an
    // inexpensive operation and does not traverse the JIT tree.
    if (root_node->getHeight() > 6 || isBufferLimit) {
        // The size of the parameters without any extra arguments from the
        // JIT tree. This includes one output Param object and 4 integers.
        constexpr size_t base_param_size =
            sizeof(Param<T>) + (4 * sizeof(uint));

        // This is the maximum size of the params that can be allowed by the
        // CUDA platform.
        constexpr size_t max_param_size = 4096 - base_param_size;

        struct tree_info {
            size_t total_buffer_size;
            size_t num_buffers;
            size_t param_scalar_size;
        };
        NodeIterator<> end_node;
        tree_info info =
            accumulate(NodeIterator<>(root_node), end_node, tree_info{0, 0, 0},
                       [](tree_info &prev, const Node &node) {
                           if (node.isBuffer()) {
                               const auto &buf_node =
                                   static_cast<const BufferNode<T> &>(node);
                               // getBytes returns the size of the data Array.
                               // Sub arrays will be represented by their parent
                               // size.
                               prev.total_buffer_size += buf_node.getBytes();
                               prev.num_buffers++;
                           } else {
                               prev.param_scalar_size += node.getParamBytes();
                           }
                           return prev;
                       });
        size_t param_size =
            info.num_buffers * sizeof(Param<T>) + info.param_scalar_size;

        // TODO: the buffer_size check here is very conservative. It
        // will trigger an evaluation of the node in most cases. We
        // should be checking the amount of memory available to guard
        // this eval
        if (param_size >= max_param_size ||
            info.total_buffer_size * 2 > lock_bytes) {
            return false;
        }
    }
    return true;
}

template<typename T>
Array<T> createNodeArray(const dim4 &dims, Node_ptr node) {
    Array<T> out = Array<T>(dims, node);
    return out;
}

template<typename T>
Array<T> createHostDataArray(const dim4 &dims, const T *const data) {
    bool is_device   = false;
    bool copy_device = false;
    return Array<T>(dims, data, is_device, copy_device);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &dims, void *data) {
    bool is_device   = true;
    bool copy_device = false;
    return Array<T>(dims, static_cast<T *>(data), is_device, copy_device);
}

template<typename T>
Array<T> createValueArray(const dim4 &dims, const T &value) {
    return createScalarNode<T>(dims, value);
}

template<typename T>
Array<T> createEmptyArray(const dim4 &dims) {
    return Array<T>(dims);
}

template<typename T>
Array<T> createSubArray(const Array<T> &parent,
                        const std::vector<af_seq> &index, bool copy) {
    parent.eval();

    dim4 dDims          = parent.getDataDims();
    dim4 dStrides       = calcStrides(dDims);
    dim4 parent_strides = parent.strides();

    if (dStrides != parent_strides) {
        const Array<T> parentCopy = copyArray(parent);
        return createSubArray(parentCopy, index, copy);
    }

    dim4 pDims   = parent.dims();
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
Array<T> createParamArray(Param<T> &tmp, bool owner) {
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

    T *ptr = arr.get();

    CUDA_CHECK(cudaMemcpyAsync(ptr, data, bytes, cudaMemcpyHostToDevice,
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    return;
}

template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }

    T *ptr = arr.get();

    CUDA_CHECK(cudaMemcpyAsync(ptr, data, bytes, cudaMemcpyDeviceToDevice,
                               cuda::getActiveStream()));

    return;
}

template<typename T>
void Array<T>::setDataDims(const dim4 &new_dims) {
    modDims(new_dims);
    data_dims = new_dims;
    if (node->isBuffer()) { node = bufferNodePtr<T>(); }
}

#define INSTANTIATE(T)                                                        \
    template Array<T> createHostDataArray<T>(const dim4 &size,                \
                                             const T *const data);            \
    template Array<T> createDeviceDataArray<T>(const dim4 &size, void *data); \
    template Array<T> createValueArray<T>(const dim4 &size, const T &value);  \
    template Array<T> createEmptyArray<T>(const dim4 &size);                  \
    template Array<T> createParamArray<T>(Param<T> & tmp, bool owner);        \
    template Array<T> createSubArray<T>(                                      \
        const Array<T> &parent, const std::vector<af_seq> &index, bool copy); \
    template void destroyArray<T>(Array<T> * A);                              \
    template Array<T> createNodeArray<T>(const dim4 &size,                    \
                                         common::Node_ptr node);              \
    template Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset,   \
                             const T *const in_data, bool is_device);         \
    template Array<T>::Array(af::dim4 dims, const T *const in_data,           \
                             bool is_device, bool copy_device);               \
    template Array<T>::~Array();                                              \
    template Node_ptr Array<T>::getNode() const;                              \
    template void Array<T>::eval();                                           \
    template void Array<T>::eval() const;                                     \
    template T *Array<T>::device();                                           \
    template void writeHostDataArray<T>(Array<T> & arr, const T *const data,  \
                                        const size_t bytes);                  \
    template void writeDeviceDataArray<T>(                                    \
        Array<T> & arr, const void *const data, const size_t bytes);          \
    template void evalMultiple<T>(std::vector<Array<T> *> arrays);            \
    template bool passesJitHeuristics<T>(Node * n);                           \
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
INSTANTIATE(common::half)

}  // namespace cuda
