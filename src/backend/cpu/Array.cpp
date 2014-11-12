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
#include <iostream>
#include <TNJ/BufferNode.hpp>
#include <TNJ/ScalarNode.hpp>

namespace cpu
{

    using af::dim4;

    template<typename T>
    Array<T>::Array(dim4 dims):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(new T[dims.elements()]),
        parent(nullptr), node(nullptr), ready(true)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, const T * const in_data):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(new T[dims.elements()]),
        parent(nullptr), node(nullptr), ready(true)
    {
        std::copy(in_data, in_data + dims.elements(), data.get());
    }


    template<typename T>
    Array<T>::Array(af::dim4 dims, TNJ::Node *n) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(),
        parent(nullptr), node(n), ready(false)
    {
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(),
        parent(&parnt), node(nullptr), ready(true)
    { }

    template<typename T>
    Array<T>::~Array()
    { }

    using TNJ::BufferNode;
    using TNJ::Node;

    template<typename T>
    Node* Array<T>::getNode() const
    {
        if (node == NULL) {
            dim_type strs[] = {strides()[0], strides()[1], strides()[2], strides()[3]};
            BufferNode<T> *buf_node = new BufferNode<T>(get(), strs);
            const_cast<Array<T> *>(this)->node = reinterpret_cast<Node *>(buf_node);
        }

        return node;
    }

    template<typename T>
    Array<T> *
    createHostDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T> *
    createDeviceDataArray(const dim4 &size, const void *data)
    {
        Array<T> *out = new Array<T>(size, (const T * const) data);
        return out;
    }

    template<typename T>
    Array<T> *
    createValueArray(const dim4 &size, const T& value)
    {
        TNJ::ScalarNode<T> *node = new TNJ::ScalarNode<T>(value);
        return createNodeArray<T>(size, reinterpret_cast<TNJ::Node *>(node));
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
    createNodeArray(const dim4 &dims, Node *node)
    {
        return new Array<T>(dims, node);
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        Array<T> *out = new Array<T>(parent, dims, offset, stride);
        // FIXME: check what is happening with the references here
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) out = copyArray(*out);
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
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value)
    {
        Array<outType> *ret = createValueArray<outType>(dims, default_value);

        copy<inType, outType>(*ret, in, outType(default_value), 1.0);

        return ret;
    }

    template<typename T>
    void scaleArray(Array<T> &arr, double factor)
    {
        T * src_ptr = arr.get();
        for(dim_type i=0; i< (dim_type)arr.elements(); ++i)
            src_ptr[i] *= factor;
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

        data = std::shared_ptr<T>(new T[elements()]);
        T *ptr = data.get();

        dim4 ostrs = strides();
        dim4 odims = dims();

        for (int w = 0; w < odims[3]; w++) {
            dim_type offw = w * ostrs[3];

            for (int z = 0; z < odims[2]; z++) {
                dim_type offz = z * ostrs[2] + offw;

                for (int y = 0; y < odims[1]; y++) {
                    dim_type offy = y * ostrs[1] + offz;

                    for (int x = 0; x < odims[0]; x++) {
                        dim_type id = x + offy;

                        ptr[id] = *(T *)node->calc(x, y, z, w);
                    }
                }
            }
        }


        ready = true;
        // Replace the current node in any JIT possible trees with the new BufferNode
        Node *prev = node;
        node = nullptr;
        prev->reset();
        prev->replace(getNode());
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
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createRefArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createNodeArray<T>   (const dim4 &size, TNJ::Node *node); \
    template       void       scaleArray<T>       (Array<T> &arr, double factor); \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template       TNJ::Node* Array<T>::getNode() const;                \
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
    template Array<float  >* createPaddedArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value); \
    template Array<double >* createPaddedArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value); \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value); \
    template Array<int    >* createPaddedArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value); \
    template Array<uint   >* createPaddedArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value); \
    template Array<uchar  >* createPaddedArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value); \
    template Array<char   >* createPaddedArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value);

    INSTANTIATE_CREATE_PADDED_ARRAY(float )
    INSTANTIATE_CREATE_PADDED_ARRAY(double)
    INSTANTIATE_CREATE_PADDED_ARRAY(int   )
    INSTANTIATE_CREATE_PADDED_ARRAY(uint  )
    INSTANTIATE_CREATE_PADDED_ARRAY(uchar )
    INSTANTIATE_CREATE_PADDED_ARRAY(char  )

#define INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(SRC_T) \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value);

    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cfloat )
    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cdouble)

}
