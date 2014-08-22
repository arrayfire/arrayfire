#include <cassert>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <iostream>
#include <kernel/elwise.hpp> //set

using af::dim4;

namespace cuda
{
    using std::ostream;

    template<typename T>
    T* cudaMallocWrapper(const size_t &elements)
    {
        T* ptr = NULL;
        //FIXME: Add checks
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * elements);
        return ptr;
    }

    // FIXME: Add checks
    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {}

    // FIXME: Add checks
    template<typename T>
    Array<T>::Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        kernel::set(data, val, elements());
    }

    // FIXME: Add checks
    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
    ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        cudaMemcpy(data, in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice);
    }

    // FIXME: Add checks
    template<typename T>
    Array<T>::~Array() { cudaFree(data); }

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
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        assert("NOT IMPLEMENTED" && 1 != 1);
        return NULL;
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
    template       Array<T>*  createSubArray<T>       (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       void       destroyArray<T>     (const Array &A); \
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
