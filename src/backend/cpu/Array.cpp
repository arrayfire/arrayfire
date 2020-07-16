/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/Array.hpp>

#include <Param.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/traits.hpp>
#include <copy.hpp>
#include <jit/BufferNode.hpp>
#include <jit/Node.hpp>
#include <jit/ScalarNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <traits.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>

#include <algorithm>  // IWYU pragma: keep
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

using af::dim4;
using common::half;
using common::Node;
using common::Node_map_t;
using common::Node_ptr;
using common::NodeIterator;
using cpu::jit::BufferNode;
using std::adjacent_find;
using std::copy;
using std::is_standard_layout;
using std::move;
using std::vector;

namespace cpu {

template<typename T>
Node_ptr bufferNodePtr() {
    return Node_ptr(reinterpret_cast<Node *>(new BufferNode<T>()));
}

template<typename T>
Array<T>::Array(dim4 dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(memAlloc<T>(dims.elements()).release(), memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {}

template<typename T>
Array<T>::Array(const dim4 &dims, T *const in_data, bool is_device,
                bool copy_device)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data((is_device & !copy_device) ? in_data
                                      : memAlloc<T>(dims.elements()).release(),
           memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
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
    if (!is_device || copy_device) {
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + dims.elements(), data.get());
    }
}

template<typename T>
Array<T>::Array(const af::dim4 &dims, Node_ptr n)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data()
    , data_dims(dims)
    , node(move(n))
    , ready(false)
    , owner(true) {}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &strides)
    : info(parent.getDevId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(false) {}

template<typename T>
Array<T>::Array(const dim4 &dims, const dim4 &strides, dim_t offset_,
                T *const in_data, bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(is_device ? in_data : memAlloc<T>(info.total()).release(),
           memFree<T>)
    , data_dims(dims)
    , node(bufferNodePtr<T>())
    , ready(true)
    , owner(true) {
    if (!is_device) {
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + info.total(), data.get());
    }
}

template<typename T>
void Array<T>::eval() {
    if (isReady()) { return; }
    if (getQueue().is_worker()) {
        AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);
    }

    this->setId(getActiveDeviceId());

    data = shared_ptr<T>(memAlloc<T>(elements()).release(), memFree<T>);

    getQueue().enqueue(kernel::evalArray<T>, *this, this->node);
    // Reset shared_ptr
    this->node = bufferNodePtr<T>();
    ready      = true;
}

template<typename T>
void Array<T>::eval() const {
    if (isReady()) { return; }
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
T *Array<T>::device() {
    getQueue().sync();
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->get();
}

template<typename T>
void evalMultiple(vector<Array<T> *> array_ptrs) {
    vector<Array<T> *> outputs;
    vector<Node_ptr> nodes;
    vector<Param<T>> params;
    if (getQueue().is_worker()) {
        AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);
    }

    // Check if all the arrays have the same dimension
    auto it = adjacent_find(begin(array_ptrs), end(array_ptrs),
                            [](const Array<T> *l, const Array<T> *r) {
                                return l->dims() != r->dims();
                            });

    // If they are not the same. eval individually
    if (it != end(array_ptrs)) {
        for (auto ptr : array_ptrs) { ptr->eval(); }
        return;
    }

    for (Array<T> *array : array_ptrs) {
        if (array->ready) { continue; }

        array->setId(getActiveDeviceId());
        array->data =
            shared_ptr<T>(memAlloc<T>(array->elements()).release(), memFree<T>);

        outputs.push_back(array);
        params.push_back(*array);
        nodes.push_back(array->node);
    }

    if (!outputs.empty()) {
        getQueue().enqueue(kernel::evalMultiple<T>, params, nodes);
        for (Array<T> *array : outputs) {
            array->ready = true;
            array->node  = bufferNodePtr<T>();
        }
    }
}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node->isBuffer()) {
        auto *bufNode  = reinterpret_cast<BufferNode<T> *>(node.get());
        unsigned bytes = this->getDataDims().elements() * sizeof(T);
        bufNode->setData(data, bytes, getOffset(), dims().get(),
                         strides().get(), isLinear());
    }
    return node;
}

template<typename T>
Node_ptr Array<T>::getNode() const {
    if (node->isBuffer()) { return const_cast<Array<T> *>(this)->getNode(); }
    return node;
}

template<typename T>
Array<T> createHostDataArray(const dim4 &dims, const T *const data) {
    return Array<T>(dims, const_cast<T *>(data), false);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &dims, void *data) {
    return Array<T>(dims, static_cast<T *>(data), true);
}

template<typename T>
Array<T> createValueArray(const dim4 &dims, const T &value) {
    auto *node = new jit::ScalarNode<T>(value);
    return createNodeArray<T>(dims, Node_ptr(node));
}

template<typename T>
Array<T> createEmptyArray(const dim4 &dims) {
    return Array<T>(dims);
}

template<typename T>
kJITHeuristics passesJitHeuristics(Node *root_node) {
    if (!evalFlag()) { return kJITHeuristics::Pass; }
    if (root_node->getHeight() >= static_cast<int>(getMaxJitSize())) {
        return kJITHeuristics::TreeHeight;
    }

    // Check if approaching the memory limit
    if (getMemoryPressure() >= getMemoryPressureThreshold()) {
        NodeIterator<Node> it(root_node);
        NodeIterator<Node> end_node;
        size_t bytes = accumulate(it, end_node, size_t(0),
                                  [=](const size_t prev, const Node &n) {
                                      // getBytes returns the size of the data
                                      // Array. Sub arrays will be represented
                                      // by their parent size.
                                      return prev + n.getBytes();
                                  });

        if (jitTreeExceedsMemoryPressure(bytes)) {
            return kJITHeuristics::MemoryPressure;
        }
    }
    return kJITHeuristics::Pass;
}

template<typename T>
Array<T> createNodeArray(const dim4 &dims, Node_ptr node) {
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

    const dim4 &pDims = parent.dims();
    dim4 dims         = toDims(index, pDims);
    dim4 strides      = toStride(index, dDims);

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
void destroyArray(Array<T> *A) {
    delete A;
}

template<typename T>
void writeHostDataArray(Array<T> &arr, const T *const data,
                        const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }
    arr.eval();
    // Ensure the memory being written to isnt used anywhere else.
    getQueue().sync();
    memcpy(arr.get(), data, bytes);
}

template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }
    memcpy(arr.get(), static_cast<const T *const>(data), bytes);
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
    template Array<T> createSubArray<T>(                                      \
        const Array<T> &parent, const vector<af_seq> &index, bool copy);      \
    template void destroyArray<T>(Array<T> * A);                              \
    template Array<T> createNodeArray<T>(const dim4 &dims, Node_ptr node);    \
    template void Array<T>::eval();                                           \
    template void Array<T>::eval() const;                                     \
    template T *Array<T>::device();                                           \
    template Array<T>::Array(const af::dim4 &dims, T *const in_data,          \
                             bool is_device, bool copy_device);               \
    template Array<T>::Array(const af::dim4 &dims, const af::dim4 &strides,   \
                             dim_t offset, T *const in_data, bool is_device); \
    template Node_ptr Array<T>::getNode();                                    \
    template Node_ptr Array<T>::getNode() const;                              \
    template void writeHostDataArray<T>(Array<T> & arr, const T *const data,  \
                                        const size_t bytes);                  \
    template void writeDeviceDataArray<T>(                                    \
        Array<T> & arr, const void *const data, const size_t bytes);          \
    template void evalMultiple<T>(vector<Array<T> *> arrays);                 \
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

}  // namespace cpu
