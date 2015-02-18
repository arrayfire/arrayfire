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
#include <scalar.hpp>
#include <JIT/BufferNode.hpp>
#include <err_opencl.hpp>
#include <memory.hpp>

using af::dim4;

namespace opencl
{
    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(ArrayInfo::elements() * sizeof(T)), bufferFree),
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
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(ArrayInfo::elements()*sizeof(T)), bufferFree),
        node(), ready(true), offset(0), owner(true)
    {
        getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0, sizeof(T)*ArrayInfo::elements(), in_data);
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, cl_mem mem) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(new cl::Buffer(mem), bufferFree),
        node(), ready(true), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim4 &offsets, const dim4 &stride) :
        ArrayInfo(dims, offsets, stride, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()), node(), ready(true),
        offset(parent.getOffset() + calcOffset(parent.strides(), offsets)),
        owner(false)
    { }


    template<typename T>
    Array<T>::Array(Param &tmp) :
        ArrayInfo(af::dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2], tmp.info.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.info.strides[0], tmp.info.strides[1],
                           tmp.info.strides[2], tmp.info.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.data, bufferFree),
        node(), ready(true), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::~Array()
    { }

    using JIT::BufferNode;
    using JIT::Node;
    using JIT::Node_ptr;

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {
            bool is_linear = isOwner() || (this->ndims() == 1);
            BufferNode *buf_node = new BufferNode(dtype_traits<T>::getName(),
                                                  shortname<T>(true), *this, is_linear, data);
            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    using af::dim4;

    template<typename T>
    Array<T> *
    createNodeArray(const dim4 &dims, Node_ptr node)
    {
        return new Array<T>(dims, node);
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {

        Array<T> *out = new Array<T>(parent, dims, offset, stride);

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

    template<typename T>
    Array<T> *
    createHostDataArray(const dim4 &size, const T * const data)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T> *
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        Array<T> *out = new Array<T>(size, (cl_mem)(data));
        return out;
    }

    template<typename T>
    Array<T>*
    createValueArray(const dim4 &size, const T& value)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T>*
    createEmptyArray(const dim4 &size)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        Array<T> *out = new Array<T>(size);
        return out;
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
    createParamArray(Param &tmp)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
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

        data = Buffer_ptr(bufferAlloc(elements() * sizeof(T)), bufferFree);

        // Do not replace this with cast operator
        KParam info = {{dims()[0], dims()[1], dims()[2], dims()[3]},
                       {strides()[0], strides()[1], strides()[2], strides()[3]},
                       0};

        Param res = {data.get(), info};

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
    template       Array<T>*  createParamArray<T> (Param &tmp);         \
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, \
                                                   const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createRefArray<T>   (const Array<T> &parent, const dim4 &dims, \
                                                   const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createNodeArray<T>   (const dim4 &size, JIT::Node_ptr node); \
    template       JIT::Node_ptr Array<T>::getNode() const;             \
    template       void Array<T>::eval();                               \
    template       void Array<T>::eval() const;                         \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template                  Array<T>::~Array();

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
