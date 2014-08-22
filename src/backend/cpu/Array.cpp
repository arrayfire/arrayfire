#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <iostream>

namespace cpu
{
    using af::dim4;

    template<typename T>
    Array<T>::Array(dim4 dims):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(dims.elements()),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, T val):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(dims.elements(), val),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, const T * const in_data):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(in_data, in_data + dims.elements()),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(0),
        parent(&parnt)
    { }

    template<typename T>
    Array<T>::~Array()
    { }

    template<typename T>
    Array<T> &
    getWritableArray(const af_array &arr)
    {
        const Array<T> &out = getArray<T>(arr);
        return const_cast<Array<T>&>(out);
    }

    template<typename T>
    const Array<T> &
    getArray(const af_array &arr)
    {
        Array<T> *out = reinterpret_cast<Array<T>*>(arr);
        return *out;
    }

    template<typename T>
    af_array
    getHandle(const Array<T> &arr)
    {
        af_array out = reinterpret_cast<af_array>(&arr);
        return out;
    }

    template<typename T>
    Array<T> *
    createDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T> *
    createValueArray(const dim4 &size, const T& value)
    {
        Array<T> *out = new Array<T>(size, value);
        return out;
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
        // FIXME: check what is happening with the references here
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) out = copyArray(*out);
        return out;
    }

    template<typename T>
    void
    destroyArray(const af_array& arr)
    {
        Array<T> &obj = getWritableArray<T>(arr);
        delete &obj;
    }

#define INSTANTIATE(T)                                                  \
    template       Array<T>&  getWritableArray<T> (const af_array &arr); \
    template const Array<T>&  getArray<T>         (const af_array &arr); \
    template       af_array   getHandle<T>        (const Array<T> &A);  \
    template       Array<T>*  createDataArray<T>  (const dim4 &size, const T * const data); \
    template       Array<T>*  createValueArray<T> (const dim4 &size, const T &value); \
    template       Array<T>*  createEmptyArray<T> (const dim4 &size);   \
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       void       destroyArray<T>     (const af_array &arr); \
    template                  Array<T>::~Array();

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
