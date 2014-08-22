#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <kernel/set.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include "traits.hpp"
#include <backend.hpp>

namespace opencl
{
    using kernel::set;

template<typename T>
class Array : public ArrayInfo
{
    cl::Buffer  data;
    Array*      parent;

public:
    Array(af::dim4 dims);
    explicit Array(af::dim4 dims, T val);
    explicit Array(af::dim4 dims, const T * const in_data);
    ~Array();

    bool isOwner() const { return parent == nullptr; }

            cl::Buffer& get()        {  return data; }
    const   cl::Buffer& get() const  {  return data; }

};

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

template<typename T>
Array<T> *
createSubArray(const Array<T>& parent, const af::dim4 &dims, const af::dim4 &offset, const af::dim4 &stride);

template<typename T>
void
destroyArray(Array<T> &A);
}
