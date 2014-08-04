#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include <kernel/set.hpp>
#include <cl.hpp>
#include <ctx.hpp>

namespace opencl
{
    using kernel::set;

template<typename T>
class Array : public ArrayInfo
{
    cl::Buffer  data;
    Array*      parent;

public:
    bool isOwner() { return parent == nullptr; }
    Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)af::dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
    }

    explicit Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)af::dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
        set(data, val, elements());
    }

    explicit Array(af::dim4 dims, const T * const in_data) :
        ArrayInfo(dims, af::dim4(0), af::dim4(0), (af_dtype)af::dtype_traits<T>::af_type),
        data(getCtx(0), CL_MEM_READ_WRITE, ArrayInfo::elements()*sizeof(T)),
        parent()
    {
        cl::copy(getQueue(0), in_data, in_data + dims.elements(), data);
    }

    cl::Buffer& get()        {  return data; }
    const   cl::Buffer& get() const  {  return data; }

    ~Array() { }
};

// Returns a reference to a Array object. This reference is not const.
template<typename T>
const Array<T> &
getArray(const af_array &arr);

// Returns a reference to a Array object. This reference is not const.
template<typename T>
Array<T> &
getWritableArray(const af_array &arr);

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

template<typename T>
void
copyData(T *data, const af_array &arr);
}
