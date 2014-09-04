#pragma once
#include <af/defines.h>
#include <backend.hpp>

namespace cuda
{

template<typename T>
struct Param
{
    T *ptr;
    dim_type dims[4];
    dim_type strides[4];
};

template<typename T>
class CParam
{
public:
    const T *ptr;
    dim_type dims[4];
    dim_type strides[4];

    __DH__ CParam(const T *iptr, const dim_type *idims, const dim_type *istrides) :
        ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = idims[i];
            strides[i] = istrides[i];
        }
    }

    __DH__ CParam(Param<T> &in) : ptr(in.ptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = in.dims[i];
            strides[i] = in.strides[i];
        }
    }

    __DH__ ~CParam() {}
};

}
