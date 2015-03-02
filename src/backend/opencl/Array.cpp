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

    using JIT::BufferNode;
    using JIT::Node;
    using JIT::Node_ptr;

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(ArrayInfo::elements() * sizeof(T)), bufferFree),
        data_dims(dims),
        node(), ready(true), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, JIT::Node_ptr n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        data_dims(dims),
        node(n), ready(false), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(bufferAlloc(ArrayInfo::elements()*sizeof(T)), bufferFree),
        data_dims(dims),
        node(), ready(true), offset(0), owner(true)
    {
        getQueue().enqueueWriteBuffer(*data.get(), CL_TRUE, 0, sizeof(T)*ArrayInfo::elements(), in_data);
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, cl_mem mem) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(new cl::Buffer(mem), bufferFree),
        data_dims(dims),
        node(), ready(true), offset(0), owner(true)
    {
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parent, const dim4 &dims, const dim4 &offsets, const dim4 &stride) :
        ArrayInfo(dims, offsets, stride, (af_dtype)dtype_traits<T>::af_type),
        data(parent.getData()),
        data_dims(parent.getDataDims()),
        node(),
        ready(true),
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
        data_dims(af::dim4(tmp.info.dims[0], tmp.info.dims[1], tmp.info.dims[2], tmp.info.dims[3])),
        node(), ready(true), offset(0), owner(true)
    {
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

    template<typename T>
    Array<T>::~Array()
    { }

    template<typename T>
    Node_ptr Array<T>::getNode() const
    {
        if (!node) {
            bool is_linear = isLinear();
            BufferNode *buf_node = new BufferNode(dtype_traits<T>::getName(),
                                                  shortname<T>(true), *this, is_linear, data);
            const_cast<Array<T> *>(this)->node = Node_ptr(reinterpret_cast<Node *>(buf_node));
        }

        return node;
    }

    using af::dim4;

    template<typename T>
    Array<T>
    createNodeArray(const dim4 &dims, Node_ptr node)
    {
        return Array<T>(dims, node);
    }

    template<typename T>
    Array<T> createSubArray(const Array<T>& parent,
                            const std::vector<af_seq> &index,
                            bool copy)
    {
        dim4 dDims = parent.getDataDims();
        dim4 pDims = parent.dims();

        dim4 dims   = af::toDims  (index, pDims);
        dim4 offset = af::toOffset(index, dDims);
        dim4 stride = af::toStride (index, dDims);

        Array<T> out = Array<T>(parent, dims, offset, stride);

        if (!copy) return out;

        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {

            out = copyArray(out);
        }

        return out;
    }

    template<typename T>
    Array<T>
    createHostDataArray(const dim4 &size, const T * const data)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        return Array<T>(size, data);
    }

    template<typename T>
    Array<T>
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }

        return Array<T>(size, (cl_mem)(data));
    }

    template<typename T>
    Array<T>
    createValueArray(const dim4 &size, const T& value)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        return createScalarNode<T>(size, value);
    }

    template<typename T>
    Array<T>
    createEmptyArray(const dim4 &size)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        return Array<T>(size);
    }

    template<typename T>
    Array<T> *initArray()
    {
        return new Array<T>(dim4());
    }

    template<typename T>
    Array<T>
    createParamArray(Param &tmp)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !opencl::isDoubleSupported(opencl::getActiveDeviceId())) {
            TYPE_ERROR(1, (std::is_same<T, double>::value ? f64 : c64));
        }
        return Array<T>(tmp);
    }

    template<typename T>
    void
    destroyArray(Array<T> *A)
    {
        delete A;
    }

    template<typename T>
    void evalArray(const Array<T> &A)
    {
        A.eval();
    }


#define INSTANTIATE(T)                                                  \
    template       Array<T>  createHostDataArray<T>   (const dim4 &size, const T * const data); \
    template       Array<T>  createDeviceDataArray<T> (const dim4 &size, const void *data); \
    template       Array<T>  createValueArray<T>      (const dim4 &size, const T &value); \
    template       Array<T>  createEmptyArray<T>      (const dim4 &size); \
    template       Array<T>  *initArray<T      >      ();               \
    template       Array<T>  createParamArray<T>      (Param &tmp);  \
    template       Array<T>  createSubArray<T>        (const Array<T> &parent, \
                                                       const std::vector<af_seq> &index, \
                                                       bool copy);      \
    template       void      destroyArray<T>          (Array<T> *A);    \
    template       void      evalArray<T>             (const Array<T> &A); \
    template       Array<T>  createNodeArray<T>       (const dim4 &size, JIT::Node_ptr node); \
    template       Array<T>::~Array        ();                          \
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
    INSTANTIATE(intl)
    INSTANTIATE(uintl)

}
