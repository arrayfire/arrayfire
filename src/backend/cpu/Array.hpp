//This is the array implementation class.
#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>

#include <memory>
#include <vector>
namespace cpu
{

using std::vector;
using af::dim4;

/// Array Arrayementation
// This class handles all resouces and relationships of an af_array object.
template<typename T>
class Array : public ArrayInfo
{
    //TODO: Generator based array

    //data if parent. empty if child
    std::vector<T> data;

    //If parent is valid. use offset to get values
    const Array<T> *parent;

public:

    bool isOwner() const {
        return parent == nullptr;
    }

    T *get(bool withOffset = true)
    {
        return const_cast<T*>(static_cast<const Array<T>*>(this)->get());
    }

    const   T *get(bool withOffset = true) const
    {
        const T* ptr = nullptr;
        if(parent == nullptr) {
            ptr = &data.front();
        }
        else {
            size_t offset = 0;
            if(withOffset) {
                offset = calcGlobalOffset(  *static_cast<const ArrayInfo*>(this),
                                            *static_cast<const ArrayInfo*>(parent));
            }
            ptr = &parent->data.front() + offset;
        }
        return ptr;
   }

    Array(dim4 dims):
                    ArrayInfo(dims, dim4(0,0,0,0), calcBaseStride(dims), (af_dtype)af::dtype_traits<T>::af_type),
                    data(dims.elements()),
                    parent(nullptr)
    { }

    Array(dim4 dims, T val):
                    ArrayInfo(dims, dim4(0,0,0,0), calcBaseStride(dims), (af_dtype)af::dtype_traits<T>::af_type),
                    data(dims.elements(), val),
                    parent(nullptr)
    { }

    Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)af::dtype_traits<T>::af_type),
        data(0),
        parent(&parnt)
    { }


    ~Array() {}

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
Array<T> *
createArray(const af::dim4 &size, const T& value);

// Creates a new Array View(sub array).
template<typename T>
Array<T> *
createView(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
void
deleteArray(const af_array& arr);

template<typename T>
void
operator <<(std::ostream &out, const Array<T> &arr);

}
