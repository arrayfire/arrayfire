#pragma once
#include <af/array.h>
#include <af/dim4.hpp>


/// Array Arrayementation Info class
// This class is the base class to all Array objects. The purpose of this class
// was to have a way to retrieve basic information of an Array object without
// specifying what type the object is at compile time.
class ArrayInfo
{
private:
    af_dtype        type;
    af::dim4        dim_size;
    af::dim4        dim_offsets, dim_strides;

public:
    ArrayInfo(af::dim4 size, af::dim4 offset, af::dim4 stride, af_dtype af_type):
        type(af_type),
        dim_size(size),
        dim_offsets(offset),
        dim_strides(stride)
    { }

#if __cplusplus > 199711L
    //Copy constructors are deprecated if there is a
    //user-defined destructor in c++11
    ArrayInfo(const ArrayInfo& other) = default;
#endif
    ~ArrayInfo() {}

    const af_dtype& getType() const     { return type;                  }

    const af::dim4& offsets() const     { return dim_offsets;           }

    const af::dim4& strides()    const  { return dim_strides;           }

    size_t elements() const             { return dim_size.elements();   }
    size_t ndims() const                { return dim_size.ndims();      }
    const af::dim4& dims() const        { return dim_size;              }

    void modDims(const af::dim4 &newDims);

    void modStrides(const af::dim4 &newStrides);

    bool isEmpty();

    bool isScalar();

    bool isRow();

    bool isColumn();

    bool isVector();

    bool isComplex();

    bool isReal();

    bool isDouble();

    bool isSingle();

    bool isRealFloating();

    bool isFloating();

    bool isInteger();
};

dim_type
calcOffset(const af::dim4 &strides, const af::dim4 &offsets);

af::dim4
calcStrides(const af::dim4 &parentDim);

// Returns size and time info for an array object.
// Note this doesn't require template parameters.
const  ArrayInfo&
getInfo(const af_array arr);
