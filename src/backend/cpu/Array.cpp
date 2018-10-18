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

#include <JIT/BufferNode.hpp>
#include <JIT/Node.hpp>
#include <JIT/ScalarNode.hpp>
#include <Param.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <traits.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>

#include <algorithm> // IWYU pragma: keep
#include <cstring>
#include <cstddef>
#include <type_traits>

namespace cpu
{

using JIT::BufferNode;
using JIT::Node;
using JIT::Node_ptr;
using JIT::Node_map_t;

using af::dim4;
using std::vector;
using std::is_standard_layout;
using std::copy;

template<typename T>
Node_ptr bufferNodePtr()
{
    return Node_ptr(reinterpret_cast<Node *>(new BufferNode<T>()));
}

template<typename T>
Array<T>::Array(dim4 dims):
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data(memAlloc<T>(dims.elements()).release(), memFree<T>), data_dims(dims),
    node(bufferNodePtr<T>()), ready(true), owner(true)
{ }

template<typename T>
Array<T>::Array(dim4 dims, const T * const in_data, bool is_device, bool copy_device):
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data((is_device & !copy_device) ? (T*)in_data : memAlloc<T>(dims.elements()).release(), memFree<T>), data_dims(dims),
    node(bufferNodePtr<T>()), ready(true), owner(true)
{
    static_assert(is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
    static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
    if (!is_device || copy_device) {
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + dims.elements(), data.get());
    }
}

template<typename T>
Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data(), data_dims(dims),
    node(n), ready(false), owner(true)
{
}

template<typename T>
Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim_t &offset_, const dim4 &strides) :
    info(parent.getDevId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
    data(parent.getData()), data_dims(parent.getDataDims()),
    node(bufferNodePtr<T>()),
    ready(true), owner(false)
{ }

template<typename T>
Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset_,
                const T * const in_data, bool is_device) :
    info(getActiveDeviceId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
    data(is_device ? (T*)in_data : memAlloc<T>(info.total()).release(), memFree<T>),
    data_dims(dims),
    node(bufferNodePtr<T>()),
    ready(true),
    owner(true)
{
    if (!is_device) {
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + info.total(), data.get());
    }
}

template<typename T>
void Array<T>::eval()
{
    if (isReady()) return;
    if (getQueue().is_worker()) AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);

    this->setId(getActiveDeviceId());

    data = shared_ptr<T>(memAlloc<T>(elements()).release(), memFree<T>);

    getQueue().enqueue(kernel::evalArray<T>, *this, this->node);
    // Reset shared_ptr
    this->node = bufferNodePtr<T>();
    ready = true;
}

template<typename T>
void Array<T>::eval() const
{
    if (isReady()) return;
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
T* Array<T>::device()
{
    getQueue().sync();
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->get();
}

template<typename T>
void evalMultiple(vector<Array<T>*> array_ptrs)
{
    vector<Array<T>> arrays;
    vector<JIT::Node_ptr> nodes;
    bool isWorker = getQueue().is_worker();
    for (auto &array : array_ptrs) {
        if (array->ready) continue;
        if (isWorker) AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);
        array->setId(getActiveDeviceId());
        array->data = shared_ptr<T>(memAlloc<T>(array->elements()).release(), memFree<T>);
        arrays.push_back(*array);
        nodes.push_back(array->node);
    }

    vector<Param<T>> params(arrays.begin(), arrays.end());
    if (arrays.size() > 0) {
        getQueue().enqueue(kernel::evalMultiple<T>, params, nodes);
        for (auto &array : array_ptrs) {
            if (array->ready) continue;
            array->ready = true;
            array->node = bufferNodePtr<T>();
        }
    }
    return;
}

template<typename T>
Node_ptr Array<T>::getNode() const
{
    if (node->isBuffer()) {
        BufferNode<T> *bufNode = reinterpret_cast<BufferNode<T> *>(node.get());
        unsigned bytes = this->getDataDims().elements() * sizeof(T);
        bufNode->setData(data,
                         bytes,
                         getOffset(),
                         dims().get(),
                         strides().get(),
                         isLinear());
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
    JIT::ScalarNode<T> *node = new JIT::ScalarNode<T>(value);
    return createNodeArray<T>(size, JIT::Node_ptr(
                                  reinterpret_cast<JIT::Node *>(node)));
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

                Node *n = node.get();

                Node_map_t nodes_map;
                vector<Node *> full_nodes;
                n->getNodesMap(nodes_map, full_nodes);
                unsigned length =0, buf_count = 0, bytes = 0;
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
Array<T> createSubArray(const Array<T>& parent,
                        const vector<af_seq> &index,
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
void
destroyArray(Array<T> *A)
{
    delete A;
}

template<typename T>
void
writeHostDataArray(Array<T> &arr, const T * const data, const size_t bytes)
{
    if(!arr.isOwner()) {
        arr = copyArray<T>(arr);
    }
    arr.eval();
    // Ensure the memory being written to isnt used anywhere else.
    getQueue().sync();
    memcpy(arr.get(), data, bytes);
}

template<typename T>
void
writeDeviceDataArray(Array<T> &arr, const void * const data, const size_t bytes)
{
    if(!arr.isOwner()) {
        arr = copyArray<T>(arr);
    }
    memcpy(arr.get(), (const T * const)data, bytes);
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
    template       Array<T>  createSubArray<T>        (const Array<T> &parent, \
                                                       const vector<af_seq> &index, \
                                                       bool copy);      \
    template       void      destroyArray<T>          (Array<T> *A);    \
    template       Array<T>  createNodeArray<T>       (const dim4 &size, JIT::Node_ptr node); \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       T*   Array<T>::device();                             \
    template       Array<T>::Array(af::dim4 dims, const T * const in_data, \
                                   bool is_device, bool copy_device);   \
    template       Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset, \
                                   const T * const in_data,             \
                                   bool is_device);                     \
    template       JIT::Node_ptr Array<T>::getNode() const;             \
    template       void      writeHostDataArray<T>    (Array<T> &arr, const T * const data, const size_t bytes); \
    template       void      writeDeviceDataArray<T>  (Array<T> &arr, const void * const data, const size_t bytes); \
    template       void      evalMultiple<T>     (vector<Array<T>*> arrays); \
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
