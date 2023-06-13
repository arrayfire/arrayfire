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

#include <nonstd/span.hpp>
#include <algorithm>  // IWYU pragma: keep
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

using af::dim4;
using arrayfire::common::half;
using arrayfire::common::Node;
using arrayfire::common::Node_map_t;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::cpu::jit::BufferNode;

using nonstd::span;
using std::accumulate;
using std::adjacent_find;
using std::copy;
using std::find_if;
using std::is_standard_layout;
using std::make_shared;
using std::move;
using std::vector;

namespace arrayfire {
namespace cpu {

template<typename T>
shared_ptr<BufferNode<T>> bufferNodePtr() {
    return std::make_shared<BufferNode<T>>();
}

template<typename T>
Array<T>::Array(dim4 dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(memAlloc<T>(dims.elements()).release(), memFree)
    , data_dims(dims)
    , node()
    , owner(true) {}

template<typename T>
Array<T>::Array(const dim4 &dims, T *const in_data, bool is_device,
                bool copy_device)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data((is_device & !copy_device) ? in_data
                                      : memAlloc<T>(dims.elements()).release(),
           memFree)
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
    , owner(true) {}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &strides)
    : info(parent.getDevId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node()
    , owner(false) {}

template<typename T>
Array<T>::Array(const dim4 &dims, const dim4 &strides, dim_t offset_,
                T *const in_data, bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(is_device ? in_data : memAlloc<T>(info.total()).release(), memFree)
    , data_dims(dims)
    , node()
    , owner(true) {
    if (!is_device) {
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + info.total(), data.get());
    }
}

template<typename T>
void checkAndMigrate(const Array<T> &arr) {
    return;
}

template<typename T>
void Array<T>::eval() {
    evalMultiple<T>({this});
}

template<typename T>
void Array<T>::eval() const {
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
T *Array<T>::device() {
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    getQueue().sync();
    return this->get();
}

template<typename T>
void evalMultiple(vector<Array<T> *> array_ptrs) {
    vector<Array<T> *> outputs;
    vector<common::Node_ptr> nodes;
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
        if (array->isReady()) { continue; }

        array->setId(getActiveDeviceId());
        array->data =
            shared_ptr<T>(memAlloc<T>(array->elements()).release(), memFree);

        outputs.push_back(array);
        params.emplace_back(array->getData().get(), array->dims(),
                            array->strides());
        nodes.push_back(array->node);
    }

    if (params.empty()) return;

    getQueue().enqueue(cpu::kernel::evalMultiple<T>, params, nodes);

    for (Array<T> *array : outputs) { array->node.reset(); }
}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node) { return node; }

    std::shared_ptr<BufferNode<T>> out = bufferNodePtr<T>();
    unsigned bytes = this->getDataDims().elements() * sizeof(T);
    out->setData(data, bytes, getOffset(), dims().get(), strides().get(),
                 isLinear());
    return out;
}

template<typename T>
Node_ptr Array<T>::getNode() const {
    return const_cast<Array<T> *>(this)->getNode();
}

template<typename T>
Array<T> createHostDataArray(const dim4 &dims, const T *const data) {
    return Array<T>(dims, const_cast<T *>(data), false);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &dims, void *data, bool copy) {
    bool is_device = true;
    return Array<T>(dims, static_cast<T *>(data), is_device, copy);
}

template<typename T>
Array<T> createValueArray(const dim4 &dims, const T &value) {
    return createNodeArray<T>(dims, make_shared<jit::ScalarNode<T>>(value));
}

template<typename T>
Array<T> createEmptyArray(const dim4 &dims) {
    return Array<T>(dims);
}

template<typename T>
kJITHeuristics passesJitHeuristics(span<Node *> root_nodes) {
    if (!evalFlag()) { return kJITHeuristics::Pass; }
    size_t bytes = 0;
    for (Node *n : root_nodes) {
        if (n->getHeight() > static_cast<int>(getMaxJitSize())) {
            return kJITHeuristics::TreeHeight;
        }
        // Check if approaching the memory limit
        if (getMemoryPressure() >= getMemoryPressureThreshold()) {
            NodeIterator<Node> it(n);
            NodeIterator<Node> end_node;
            bytes = accumulate(it, end_node, bytes,
                               [=](const size_t prev, const Node &n) {
                                   // getBytes returns the size of the data
                                   // Array. Sub arrays will be represented
                                   // by their parent size.
                                   return prev + n.getBytes();
                               });
        }
    }

    if (jitTreeExceedsMemoryPressure(bytes)) {
        return kJITHeuristics::MemoryPressure;
    }

    return kJITHeuristics::Pass;
}

template<typename T>
Array<T> createNodeArray(const dim4 &dims, Node_ptr node) {
    Array<T> out(dims, node);
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
    data_dims = new_dims;
    modDims(new_dims);
}

#define INSTANTIATE(T)                                                        \
    template Array<T> createHostDataArray<T>(const dim4 &dims,                \
                                             const T *const data);            \
    template Array<T> createDeviceDataArray<T>(const dim4 &dims, void *data,  \
                                               bool copy);                    \
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
    template kJITHeuristics passesJitHeuristics<T>(span<Node *> n);           \
    template void Array<T>::setDataDims(const dim4 &new_dims);                \
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

}  // namespace cpu
}  // namespace arrayfire
