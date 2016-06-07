/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <err_common.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <kernel/Array.hpp>
#include <TNJ/BufferNode.hpp>
#include <TNJ/ScalarNode.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <cstring>
#include <cstddef>
#include <MemoryManager.hpp>

namespace cpu
{

const int MAX_TNJ_LEN = 20;
using TNJ::BufferNode;
using TNJ::Node;
using TNJ::Node_ptr;

using af::dim4;

template<typename T>
Array<T>::Array(dim4 dims):
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data(memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
    node(), ready(true), owner(true)
{ }

template<typename T>
Array<T>::Array(dim4 dims, const T * const in_data, bool is_device, bool copy_device):
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data((is_device & !copy_device) ? (T*)in_data : memAlloc<T>(dims.elements()), memFree<T>), data_dims(dims),
    node(), ready(true), owner(true)
{
    static_assert(std::is_standard_layout<Array<T>>::value, "Array<T> must be a standard layout type");
    static_assert(offsetof(Array<T>, info) == 0, "Array<T>::info must be the first member variable of Array<T>");
    if (!is_device || copy_device) {
        std::copy(in_data, in_data + dims.elements(), data.get());
    }
}

template<typename T>
Array<T>::Array(af::dim4 dims, TNJ::Node_ptr n) :
    info(getActiveDeviceId(), dims, 0, calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
    data(), data_dims(dims),
    node(n), ready(false), owner(true)
{
}

template<typename T>
Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim_t &offset_, const dim4 &strides) :
    info(parent.getDevId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
    data(parent.getData()), data_dims(parent.getDataDims()),
    node(),
    ready(true), owner(false)
{ }

template<typename T>
Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset_,
                const T * const in_data, bool is_device) :
    info(getActiveDeviceId(), dims, offset_, strides, (af_dtype)dtype_traits<T>::af_type),
    data(is_device ? (T*)in_data : memAlloc<T>(info.total()), memFree<T>),
    data_dims(dims),
    node(),
    ready(true),
    owner(true)
{
    if (!is_device) {
        std::copy(in_data, in_data + info.total(), data.get());
    }
}

template<typename T>
void Array<T>::eval()
{
    if (isReady()) return;
    if (getQueue().is_worker()) AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);

    this->setId(getActiveDeviceId());

    data = std::shared_ptr<T>(memAlloc<T>(elements()), memFree<T>);

    getQueue().enqueue(kernel::evalArray<T>, *this);

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
void evalMultiple(std::vector<Array<T>*> arrays)
{
    //FIXME: implement this correctly
    //Using fallback for now
    for (auto array : arrays) {
        array->eval();
    }
    return;
}

template<typename T>
Node_ptr Array<T>::getNode() const
{
    if (!node) {

        unsigned bytes = this->getDataDims().elements() * sizeof(T);

        BufferNode<T> *buf_node = new BufferNode<T>(data,
                                                    bytes,
                                                    getOffset(),
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

    if (evalFlag()) {
        unsigned length =0, buf_count = 0, bytes = 0;

        Node *n = node.get();
        n->getInfo(length, buf_count, bytes);
        n->reset();

        if (length > getMaxJitSize() ||
            buf_count >= getMaxBuffers() ||
            bytes >= getMaxBytes()) {
            out.eval();
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
    template       Array<T>  createNodeArray<T>       (const dim4 &size, TNJ::Node_ptr node); \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       Array<T>::Array(af::dim4 dims, const T * const in_data, \
                                   bool is_device, bool copy_device);   \
    template       Array<T>::Array(af::dim4 dims, af::dim4 strides, dim_t offset, \
                                   const T * const in_data,             \
                                   bool is_device);                     \
    template       TNJ::Node_ptr Array<T>::getNode() const;             \
    template       void      writeHostDataArray<T>    (Array<T> &arr, const T * const data, const size_t bytes); \
    template       void      writeDeviceDataArray<T>  (Array<T> &arr, const void * const data, const size_t bytes); \
    template       void      evalMultiple<T>     (std::vector<Array<T>*> arrays); \

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
