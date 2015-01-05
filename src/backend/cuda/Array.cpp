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
#include <stdexcept>
#include <copy.hpp>
#include <err_cuda.hpp>
#include <JIT/BufferNode.hpp>
#include <scalar.hpp>
#include <memory.hpp>

using af::dim4;

namespace cuda
{

    using std::ostream;

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(memAlloc<T>(dims.elements()), memFree<T>),
        node(), ready(true), offset(0), owner(true)
    {}

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data, bool is_device) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data((is_device ? (T *)in_data : memAlloc<T>(dims.elements())), memFree<T>),
        node(), ready(true), offset(0), owner(true)
    {
        if (!is_device) {
            CUDA_CHECK(cudaMemcpy(data.get(), in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim4 &offsets, const dim4 &strides) :
        ArrayInfo(dims, offsets, strides, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()),
        node(), ready(true),
        offset(parent.getOffset() + calcOffset(parent.strides(), offsets)),
        owner(false)
    { }

    template<typename T>
    Array<T>::Array(Param<T> &tmp) :
        ArrayInfo(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2], tmp.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.ptr, memFree<T>),
        node(), ready(true), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        node(n), ready(false), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::~Array() {}


    using JIT::BufferNode;
    using JIT::Node;
    using JIT::Node_ptr;

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {
            bool is_linear = isOwner() || (this->ndims() == 1);
            BufferNode<T> *buf_node = new BufferNode<T>(irname<T>(),
                                                        shortname<T>(true), data,
                                                        *this, offset, is_linear);
            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    template<typename T>
    Array<T> *
    createNodeArray(const dim4 &dims, Node_ptr node)
    {
        return new Array<T>(dims, node);
    }

    template<typename T>
    Array<T> *
    createHostDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data, false);
        return out;
    }

    template<typename T>
    Array<T> *
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        Array<T> *out = new Array<T>(size, (const T * const)data, true);
        return out;
    }

    template<typename T>
    Array<T>*
    createValueArray(const dim4 &size, const T& value)
    {
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T>*
    createEmptyArray(const dim4 &size)
    {
        Array<T> *out = new Array<T>(size);
        return out;
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {

        Array<T> *out = new Array<T>(parent, dims, offset, stride);

        // FIXME: Implement this for CUDA
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {
            out = copyArray(*out);
        }

        return out;
    }

    template<typename T>
    Array<T> *
    createRefArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        return new Array<T>(parent, dims, offset, stride);
    }

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value, double factor)
    {
        Array<outType> *ret = createEmptyArray<outType>(dims);

        copy<inType, outType>(*ret, in, default_value, factor);

        return ret;
    }

    template<typename T>
    Array<T>*
    createParamArray(Param<T> &tmp)
    {
        Array<T> *out = new Array<T>(tmp);
        return out;
    }

    template<typename T>
    void
    destroyArray(Array<T> &A)
    {
        delete &A;
    }

    template<typename T>
    void Array<T>::eval()
    {
        if (isReady()) return;

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

#define INSTANTIATE(T)                                                  \
    template       Array<T>*  createHostDataArray<T>  (const dim4 &size, const T * const data); \
    template       Array<T>*  createDeviceDataArray<T>  (const dim4 &size, const void *data); \
    template       Array<T>*  createValueArray<T> (const dim4 &size, const T &value); \
    template       Array<T>*  createEmptyArray<T> (const dim4 &size);   \
    template       Array<T>*  createParamArray<T> (Param<T> &tmp);      \
    template       Array<T>*  createSubArray<T>       (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createRefArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template       Array<T>*  createNodeArray<T>   (const dim4 &size, JIT::Node_ptr node); \
    template                  Array<T>::~Array();                       \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#define INSTANTIATE_CREATE_PADDED_ARRAY(SRC_T) \
    template Array<float  >* createPaddedArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double >* createPaddedArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    >* createPaddedArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   >* createPaddedArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<uchar  >* createPaddedArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   >* createPaddedArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor);

    INSTANTIATE_CREATE_PADDED_ARRAY(float )
    INSTANTIATE_CREATE_PADDED_ARRAY(double)
    INSTANTIATE_CREATE_PADDED_ARRAY(int   )
    INSTANTIATE_CREATE_PADDED_ARRAY(uint  )
    INSTANTIATE_CREATE_PADDED_ARRAY(uchar )
    INSTANTIATE_CREATE_PADDED_ARRAY(char  )

#define INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(SRC_T) \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor);

    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cfloat )
    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cdouble)

}
