//This is the array implementation class.
#pragma once
#include <af/array.h>
#include <af/dim4.hpp>
#include <ArrayInfo.hpp>
#include "traits.hpp"
#include <backend.hpp>

#include <memory>
#include <vector>
namespace cpu
{

using std::vector;
using af::dim4;

/// Array Arrayementation
template<typename T>
class Array : public ArrayInfo
{
    //TODO: Generator based array

    //data if parent. empty if child
    std::vector<T> data;

    //If parent is valid. use offset to get values
    const Array<T> *parent;

public:

    Array(dim4 dims);
    explicit Array(dim4 dims, T val);
    explicit Array(dim4 dims, const T * const in_data);
    Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    ~Array();

    bool isOwner() const
    {
        return parent == nullptr;
    }

    T* get(bool withOffset = true)
    {
        return const_cast<T*>(static_cast<const Array<T>*>(this)->get());
    }

    const T* get(bool withOffset = true) const
    {

        const T* ptr = nullptr;
        if(parent == nullptr) {
            ptr = &data.front();
        }
        else {
            size_t offset = 0;
            if(withOffset) {
                offset = calcOffset(parent->strides(), this->offsets());
            }
            ptr = &parent->data.front() + offset;
        }
        return ptr;
    }
};

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T> *
createValueArray(const af::dim4 &size, const T& value);

// Creates a new Array object on the heap and returns a reference to it.
template<typename T>
Array<T>*
createDataArray(const af::dim4 &size, const T * const data);

// Create an Array object and do not assign any values to it
template<typename T>
Array<T>*
createEmptyArray(const af::dim4 &size);

// Creates a new Array View(sub array).
template<typename T>
Array<T> *
createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

template<typename T>
void
destroyArray(Array<T> &arr);

template<typename T>
void
operator <<(std::ostream &out, const Array<T> &arr);

}
