#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include "traits.hpp"
#include <backend.hpp>
#include <cuda_runtime_api.h>

namespace cuda
{

template<typename T>
T* cudaMallocWrapper(const size_t &elements);

template<typename T>
class Array : public ArrayInfo
{
    T*      data;
    Array*  parent;

public:

    Array(af::dim4 dims);
    explicit Array(af::dim4 dims, T val);
    explicit Array(af::dim4 dims, const T * const in_data);

    bool isOwner() const { return parent == NULL; }

            T* get()        {  return data; }
    const   T* get() const  {  return data; }

    // FIXME: Add checks
    ~Array();
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

// Create an Array object and do not assign any values to it
template<typename T>
Array<T>*
createEmptyArray(const af::dim4 &size);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
void
destroyArray(const af_array &arr);

template<typename T>
Array<T> *
createSubArray(const Array<T>& parent, const af::dim4 &dims, const af::dim4 &offset, const af::dim4 &stride);

}
