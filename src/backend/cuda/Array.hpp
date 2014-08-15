#pragma once
#include <cuda_runtime_api.h>
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include "traits.hpp"
#include "backend.h"
#include <kernel/elwise.hpp> //set

namespace cuda
{

template<typename T>
T* cudaMallocWrapper(const size_t &elements) {
    T* ptr = NULL;
    //FIXME: Add checks
    cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * elements);
    return ptr;
}

template<typename T>
class Array : public ArrayInfo
{
    T*      data;
    Array*  parent;
public:
    bool isOwner() const {
        return parent == NULL;
    }

    // FIXME: Add checks
    Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {}

    // FIXME: Add checks
    explicit Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        kernel::set(data, val, elements());
    }

    // FIXME: Add checks
    explicit Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        cudaMemcpy(data, in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice);
    }

            T* get()        {  return data; }
    const   T* get() const  {  return data; }

    // FIXME: Add checks
    ~Array() { cudaFree(data); }
};

// Returns a reference to a Array object. This reference is not const.
template<typename T>
Array<T> &
getWritableArray(const af_array &arr);

// Returns a constant reference to the Array object from an af_array object
template<typename T>
const  Array<T>&
getArray(const af_array &arr);

// Returns the af_array handle for the Array object.
template<typename T>
af_array
getHandle(const Array<T> &arr);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T>*
createValueArray(const af::dim4 &size, const T& value);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T>*
createDataArray(const af::dim4 &size, const T * const data);

template<typename T>
Array<T> *
createView(const Array<T>& parent, const af::dim4 &dims, const af::dim4 &offset, const af::dim4 &stride);
}
