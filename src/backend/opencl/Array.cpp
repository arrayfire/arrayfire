#include <af/dim4.hpp>
#include <Array.hpp>
#include <cassert>
#include <iostream>
#include <backend.hpp>
using af::dim4;

namespace opencl
{
    using std::ostream;

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
        set(data, val, elements());
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
        cl::copy(getQueue(0), in_data, in_data + dims.elements(), data);
    }

    template<typename T>
    Array<T>::~Array()
    { }

    using af::dim4;

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        assert("NOT IMPLEMENTED" && 1 != 1);
        return NULL;
    }

    template<typename T>
    Array<T> *
    createDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T>*
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
    void
    destroyArray(Array<T> &A)
    {
        delete &A;
    }

#define INSTANTIATE(T)                                                  \
    template       Array<T>*  createDataArray<T>  (const dim4 &size, const T * const data); \
    template       Array<T>*  createValueArray<T> (const dim4 &size, const T &value); \
    template       Array<T>*  createEmptyArray<T> (const dim4 &size);   \
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
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
}
