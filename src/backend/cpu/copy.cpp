#include <type_traits>
#include <af/array.h>
#include <Array.hpp>
#include <copy.hpp>
#include <cstring>

#include <complex>
#include <vector>
#include <cassert>

namespace cpu
{

template<typename T>
void strided_copy(T* dst, const T* src, const dim4 &dims, const dim4 &strides, unsigned dim)
{
    if(dim == 0) {
        if(strides[dim] == 1) {
            //FIXME: Check for errors / exceptions
            memcpy(dst, src, dims[dim] * sizeof(T));
        } else {
            for(int i = 0; i < dims[dim]; i++) {
                dst[i] = src[strides[dim]*i];
            }
        }
    } else {
        for(int i = dims[dim]; i > 0; i--) {
            strided_copy<T>(dst, src, dims, strides, dim - 1);
            src += strides[dim];
            dst += dims[dim-1];
        }
    }
}

// Assigns to single elements
template<typename T>
void assignVal(T *to, const Array<T> &from, size_t elements)
{
    if(from.isOwner()) {
        // FIXME: Check for errors / exceptions
        memcpy(to, from.get(), elements*sizeof(T));
    } else {
        strided_copy<T>(to, from.get(), from.dims(), from.strides(), from.ndims() - 1);
    }
}

template<typename T>
void copyData(T *data, const af_array &arr)
{
    const Array<T> &val_arr = getArray<T>(arr);
    assignVal(data, val_arr, val_arr.elements()); //TODO: Do a simple memcpy for the cpu version
    return;
}

template void copyData<float>(float *data, const af_array &dst);
template void copyData<cfloat>(cfloat *data, const af_array &dst);
template void copyData<double>(double *data, const af_array &dst);
template void copyData<cdouble>(cdouble *data, const af_array &dst);
template void copyData<char>(char *data, const af_array &dst);
template void copyData<int>(int *data, const af_array &dst);
template void copyData<unsigned>(unsigned *data, const af_array &dst);
template void copyData<unsigned char>(unsigned char *data, const af_array &dst);

}
