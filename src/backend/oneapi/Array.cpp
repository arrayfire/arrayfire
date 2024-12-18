/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

#include <Param.hpp>
#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/half.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/jit/ScalarNode.hpp>
#include <common/util.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <jit/BufferNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <scalar.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <numeric>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <vector>

using af::dim4;
using af::dtype_traits;

using arrayfire::common::half;
using arrayfire::common::Node;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::oneapi::jit::BufferNode;

using nonstd::span;
using std::accumulate;
using std::is_standard_layout;
using std::make_shared;
using std::shared_ptr;
using std::vector;

using sycl::buffer;

namespace arrayfire {
namespace oneapi {
namespace {
template<typename T>
shared_ptr<BufferNode<T>> bufferNodePtr() {
    return make_shared<BufferNode<T>>(
        static_cast<af::dtype>(dtype_traits<T>::af_type));
}

template<typename T>
void verifyTypeSupport() {}

template<>
void verifyTypeSupport<double>() {
    if (!isDoubleSupported(getActiveDeviceId())) {
        AF_ERROR("Double precision not supported", AF_ERR_NO_DBL);
    }
}

template<>
void verifyTypeSupport<cdouble>() {
    if (!isDoubleSupported(getActiveDeviceId())) {
        AF_ERROR("Double precision not supported", AF_ERR_NO_DBL);
    }
}

template<>
void verifyTypeSupport<arrayfire::common::half>() {
    if (!isHalfSupported(getActiveDeviceId())) {
        AF_ERROR("Half precision not supported", AF_ERR_NO_HALF);
    }
}
}  // namespace

template<typename T>
void checkAndMigrate(const Array<T> &arr) {
    if (arr.getDevId() != detail::getActiveDeviceId()) {
        AF_ERROR("Input Array not created on current device", AF_ERR_DEVICE);
    }
}

template<typename T>
Array<T>::Array(const dim4 &dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(memAlloc<T>(info.elements()).release(), memFree<T>)
    , data_dims(dims)
    , node()
    , owner(true) {}

template<typename T>
Array<T>::Array(const dim4 &dims, Node_ptr n)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data_dims(dims)
    , node(std::move(n))
    , owner(true) {
    if (node->isBuffer()) {
        data = std::static_pointer_cast<BufferNode<T>>(node)->getDataPointer();
    }
}

template<typename T>
Array<T>::Array(const dim4 &dims, const T *const in_data)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(memAlloc<T>(info.elements()).release(), memFree<T>)
    , data_dims(dims)
    , node()
    , owner(true) {
    static_assert(is_standard_layout<Array<T>>::value,
                  "Array<T> must be a standard layout type");
    static_assert(std::is_nothrow_move_assignable<Array<T>>::value,
                  "Array<T> is not move assignable");
    static_assert(std::is_nothrow_move_constructible<Array<T>>::value,
                  "Array<T> is not move constructible");
    static_assert(
        offsetof(Array<T>, info) == 0,
        "Array<T>::info must be the first member variable of Array<T>");
    getQueue()
        .submit([&](sycl::handler &h) {
            h.copy(in_data, data->get_access(h, sycl::range(info.elements())));
        })
        .wait();
}

template<typename T>
Array<T>::Array(const af::dim4 &dims, buffer<T> *const mem, size_t offset,
                bool copy)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(copy ? memAlloc<T>(info.elements()).release() : new buffer<T>(*mem),
           memFree<T>)
    , data_dims(dims)
    , node()
    , owner(true) {
    if (copy) {
        getQueue()
            .submit([&](sycl::handler &h) {
                h.copy(mem->get_access(h, sycl::range(info.elements())),
                       data->get_access(h));
            })
            .wait();
    }
}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &stride)
    : info(parent.getDevId(), dims, offset_, stride,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node()
    , owner(false) {}

template<typename T>
Array<T>::Array(Param<T> &tmp, bool owner_)
    : info(getActiveDeviceId(),
           dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2],
                tmp.info.dims[3]),
           0,
           dim4(tmp.info.strides[0], tmp.info.strides[1], tmp.info.strides[2],
                tmp.info.strides[3]),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(
          tmp.data, owner_ ? memFree<T> : [](sycl::buffer<T> * /*unused*/) {})
    , data_dims(dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2],
                     tmp.info.dims[3]))
    , node()
    , owner(owner_) {}

template<typename T>
Array<T>::Array(const dim4 &dims, const dim4 &strides, dim_t offset_,
                const T *const in_data, bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data()
    , data_dims(dims)
    , node()
    , owner(true) {
    if (is_device) {
        buffer<T> *ptr;
        std::memcpy(&ptr, in_data, sizeof(buffer<T> *));
        data = make_shared<buffer<T>>(*ptr);
    } else {
        data = memAlloc<T>(info.elements());
        getQueue()
            .submit([&](sycl::handler &h) {
                h.copy(in_data, data->get_access(h, sycl::range(info.total())));
            })
            .wait();
    }
}

template<typename T>
void Array<T>::eval() {
    if (isReady()) { return; }

    this->setId(getActiveDeviceId());
    data = std::shared_ptr<sycl::buffer<T>>(
        memAlloc<T>(info.elements()).release(), memFree<T>);

    // Do not replace this with cast operator
    KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                   {strides()[0], strides()[1], strides()[2], strides()[3]},
                   0};

    Param<T> res{data.get(), info};

    evalNodes(res, getNode().get());
    node.reset();
}

template<typename T>
void Array<T>::eval() const {
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
buffer<T> *Array<T>::device() {
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->get();
}

template<typename T>
void evalMultiple(vector<Array<T> *> arrays) {
    vector<Param<T>> outputs;
    vector<Array<T> *> output_arrays;
    vector<Node *> nodes;

    // Check if all the arrays have the same dimension
    auto it = std::adjacent_find(begin(arrays), end(arrays),
                                 [](const Array<T> *l, const Array<T> *r) {
                                     return l->dims() != r->dims();
                                 });

    // If they are not the same. eval individually
    if (it != end(arrays)) {
        for (auto ptr : arrays) { ptr->eval(); }
        return;
    }

    for (Array<T> *array : arrays) {
        if (array->isReady()) { continue; }

        const ArrayInfo info = array->info;

        array->setId(getActiveDeviceId());
        array->data = std::shared_ptr<buffer<T>>(
            memAlloc<T>(info.elements()).release(), memFree<T>);

        // Do not replace this with cast operator
        KParam kInfo = {
            {info.dims()[0], info.dims()[1], info.dims()[2], info.dims()[3]},
            {info.strides()[0], info.strides()[1], info.strides()[2],
             info.strides()[3]},
            0};

        outputs.emplace_back(array->data.get(), kInfo);
        output_arrays.push_back(array);
        nodes.push_back(array->getNode().get());
    }

    evalNodes(outputs, nodes);

    for (Array<T> *array : output_arrays) { array->node.reset(); }
}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node) { return node; }

    AParam<T, sycl::access_mode::read> info = *this;
    unsigned bytes = this->dims().elements() * sizeof(T);
    auto nn        = bufferNodePtr<T>();
    nn->setData(info, data, bytes, isLinear());

    return nn;
}

template<typename T>
Node_ptr Array<T>::getNode() const {
    return const_cast<Array<T> *>(this)->getNode();
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
kJITHeuristics passesJitHeuristics(span<Node *> root_nodes) {
    if (!evalFlag()) { return kJITHeuristics::Pass; }
    static auto getLogger = [&] { return common::loggerFactory("jit"); };
    for (const Node *n : root_nodes) {
        if (n->getHeight() > static_cast<int>(getMaxJitSize())) {
            AF_TRACE(
                "JIT tree evaluated because of tree height exceeds limit: {} > "
                "{}",
                n->getHeight(), getMaxJitSize());
            return kJITHeuristics::TreeHeight;
        }
    }

    // TODO(umar): add memory based checks for JIT kernel generation
    bool isBufferLimit =
        false;  // getMemoryPressure() >= getMemoryPressureThreshold();
    // auto platform      = getActivePlatform();

    // The Apple platform can have the nvidia card or the AMD card
    // bool isIntel = platform == AFCL_PLATFORM_INTEL;

    /// Intels param_size limit is much smaller than the other platforms
    /// so we need to start checking earlier with smaller trees
    int heightCheckLimit = 3;

    // A lightweight check based on the height of the node. This is
    // an inexpensive operation and does not traverse the JIT tree.
    bool atHeightLimit =
        std::any_of(std::begin(root_nodes), std::end(root_nodes),
                    [heightCheckLimit](Node *n) {
                        return (n->getHeight() + 1 >= heightCheckLimit);
                    });

    if (atHeightLimit || isBufferLimit) {
        // This is the base parameter size if the kernel had no
        // arguments
        size_t base_param_size =
            (sizeof(T *) + sizeof(Param<T>)) * root_nodes.size() +
            (3 * sizeof(uint));

        const sycl::device &device = getDevice();
        size_t max_param_size =
            device.get_info<sycl::info::device::max_parameter_size>();
        // typical values:
        //   NVIDIA     = 4096
        //   AMD        = 3520  (AMD A10 iGPU = 1024)
        //   Intel iGPU = 1024
        max_param_size -= base_param_size;

        struct tree_info {
            size_t total_buffer_size;
            size_t num_buffers;
            size_t param_scalar_size;
        };

        tree_info info{0, 0, 0};
        for (Node *n : root_nodes) {
            NodeIterator<> it(n);
            info = accumulate(
                it, NodeIterator<>(), info, [](tree_info &prev, Node &n) {
                    if (n.isBuffer()) {
                        auto &buf_node = static_cast<BufferNode<T> &>(n);
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
        }
        isBufferLimit = jitTreeExceedsMemoryPressure(info.total_buffer_size);

        size_t param_size =
            (info.num_buffers * (sizeof(Param<T>) + sizeof(T *)) +
             info.param_scalar_size);

        bool isParamLimit = param_size >= max_param_size;

        if (isParamLimit) {
            AF_TRACE(
                "JIT tree evaluated because of kernel parameter size: {} >= {}",
                param_size, max_param_size);
            return kJITHeuristics::KernelParameterSize;
        }
        if (isBufferLimit) {
            AF_TRACE("JIT tree evaluated because of memory pressure: {}",
                     info.total_buffer_size);
            return kJITHeuristics::MemoryPressure;
        }
    }
    return kJITHeuristics::Pass;
}

template<typename T>
void *getDevicePtr(const Array<T> &arr) {
    const buffer<T> *buf = arr.device();
    return (void *)buf;
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
    dim4 parent_strides = parent.strides();

    if (parent.isLinear() == false) {
        const Array<T> parentCopy = copyArray(parent);
        return createSubArray(parentCopy, index, copy);
    }

    const dim4 &pDims = parent.dims();

    dim4 dims    = toDims(index, pDims);
    dim4 strides = toStride(index, dDims);

    // Find total offsets after indexing
    dim4 offsets = toOffset(index, pDims);
    dim_t offset = parent.getOffset();
    for (int i = 0; i < 4; i++) { offset += offsets[i] * parent_strides[i]; }

    Array<T> out = Array<T>(parent, dims, offset, strides);

    if (!copy) { return out; }

    if (strides[0] != 1 || strides[1] < 0 || strides[2] < 0 || strides[3] < 0) {
        out = copyArray(out);
    }

    return out;
}

template<typename T>
Array<T> createHostDataArray(const dim4 &dims, const T *const data) {
    verifyTypeSupport<T>();
    return Array<T>(dims, data);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &dims, void *data, bool copy) {
    verifyTypeSupport<T>();

    return Array<T>(dims, static_cast<buffer<T> *>(data), 0, copy);
}

template<typename T>
Array<T> createValueArray(const dim4 &dims, const T &value) {
    verifyTypeSupport<T>();
    return createScalarNode<T>(dims, value);
}

template<typename T>
Array<T> createEmptyArray(const dim4 &dims) {
    verifyTypeSupport<T>();
    return Array<T>(dims);
}

template<typename T>
Array<T> createParamArray(Param<T> &tmp, bool owner) {
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
    auto arr_get = arr.get();
    getQueue()
        .submit([&](sycl::handler &h) {
            auto host_acc =
                arr_get->template get_access<sycl::access_mode::write>(
                    h, sycl::range(bytes / sizeof(T)), arr.getOffset());
            h.copy(data, host_acc);
        })
        .wait();
}

template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }

    sycl::buffer<T> *dataptr =
        static_cast<sycl::buffer<T> *>(const_cast<void *>(data));
    auto arr_get = arr.get();
    getQueue().submit([&](sycl::handler &h) {
        auto src_acc = dataptr->template get_access<sycl::access_mode::read>(
            h, sycl::range(bytes / sizeof(T)));
        auto dst_acc = arr_get->template get_access<sycl::access_mode::write>(
            h, sycl::range(bytes / sizeof(T)), arr.getOffset());
        h.copy(src_acc, dst_acc);
    });
}

template<typename T>
void Array<T>::setDataDims(const dim4 &new_dims) {
    data_dims = new_dims;
    modDims(new_dims);
}

template<typename T>
size_t Array<T>::getAllocatedBytes() const {
    if (!isReady()) { return 0; }
    size_t bytes = memoryManager().allocated(data.get());
    // External device pointer
    if (bytes == 0 && data.get()) { return data_dims.elements() * sizeof(T); }
    return bytes;
}

#define INSTANTIATE(T)                                                       \
    template Array<T> createHostDataArray<T>(const dim4 &dims,               \
                                             const T *const data);           \
    template Array<T> createDeviceDataArray<T>(const dim4 &dims, void *data, \
                                               bool copy);                   \
    template Array<T> createValueArray<T>(const dim4 &dims, const T &value); \
    template Array<T> createEmptyArray<T>(const dim4 &dims);                 \
    template Array<T> createParamArray<T>(Param<T> & tmp, bool owner);       \
    template Array<T> createSubArray<T>(                                     \
        const Array<T> &parent, const vector<af_seq> &index, bool copy);     \
    template void destroyArray<T>(Array<T> * A);                             \
    template Array<T> createNodeArray<T>(const dim4 &dims, Node_ptr node);   \
    template Array<T>::Array(const dim4 &dims, const dim4 &strides,          \
                             dim_t offset, const T *const in_data,           \
                             bool is_device);                                \
    template Array<T>::Array(const dim4 &dims, buffer<T> *mem,               \
                             size_t src_offset, bool copy);                  \
    template Node_ptr Array<T>::getNode();                                   \
    template Node_ptr Array<T>::getNode() const;                             \
    template void Array<T>::eval();                                          \
    template void Array<T>::eval() const;                                    \
    template buffer<T> *Array<T>::device();                                  \
    template void writeHostDataArray<T>(Array<T> & arr, const T *const data, \
                                        const size_t bytes);                 \
    template void writeDeviceDataArray<T>(                                   \
        Array<T> & arr, const void *const data, const size_t bytes);         \
    template void evalMultiple<T>(vector<Array<T> *> arrays);                \
    template kJITHeuristics passesJitHeuristics<T>(span<Node *> node);       \
    template void *getDevicePtr<T>(const Array<T> &arr);                     \
    template void Array<T>::setDataDims(const dim4 &new_dims);               \
    template size_t Array<T>::getAllocatedBytes() const;                     \
    template void checkAndMigrate<T>(const Array<T> &arr);

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

}  // namespace oneapi
}  // namespace arrayfire
