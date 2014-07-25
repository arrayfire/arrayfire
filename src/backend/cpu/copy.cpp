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
    using std::unique_ptr;

template<typename T>
void strided_copy(T* dst, const T* src, const dim4 &dims, const dim4 &strides, unsigned dim)
{
    if(dim == 0) {
        if(strides[dim] == 1) {
            memcpy(dst, src, dims[dim] * sizeof(T));
        }
        else {
            for(int i = 0; i < dims[dim]; i++) {
                dst[i] = src[strides[dim]*i];
            }
        }
    }
    else {
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
        memcpy(to, from.get(), elements*sizeof(T));
    }
    else {
        strided_copy<T>(to, from.get(), from.dims(), from.strides(), from.ndims() - 1);
    }
}

template<typename T>
T* copyData(const af_array &arr)
{
    size_t elements = af_get_elements(arr);
    const Array<T> &val_arr = getArray<T>(arr);
    // can't use vector because we need to release ownership
    unique_ptr<T[]> out(new T[elements]);
    assignVal(out.get(), val_arr, elements); //TODO: Do a simple memcpy for the cpu version

    return out.release();
}


template<typename T>
void
copyData(af_array &dst, const T* const src)
{
    Array<T> &dstArray = getWritableArray<T>(dst);
    if(dstArray.isOwner()) {
        memcpy(dstArray.get(), src, dstArray.elements() * sizeof(T));
    }
    else {
        assert("NOT IMPLEMENTED" && 1 != 1);
    }
}

using std::complex;
using std::array;

template float*                             copyData<float>(const af_array &arr);
template complex<float>*                    copyData<complex<float>>(const af_array &arr);
template double*                            copyData<double>(const af_array &arr);
template complex<double>*                   copyData<complex<double>>(const af_array &arr);
template char*                              copyData<char>(const af_array &arr);
template int*                               copyData<int>(const af_array &arr);
template unsigned*                          copyData<unsigned>(const af_array &arr);
template unsigned char*                     copyData<unsigned char>(const af_array &arr);


template void copyData<float>(af_array &dst, const float* const src);
template void copyData<cfloat>(af_array &dst, const cfloat* const src);
template void copyData<double>(af_array &dst, const double* const src);
template void copyData<cdouble>(af_array &dst, const cdouble* const src);
template void copyData<char>(af_array &dst, const char* const src);
template void copyData<int>(af_array &dst, const int* const src);
template void copyData<unsigned>(af_array &dst, const unsigned* const src);
template void copyData<unsigned char>(af_array &dst, const unsigned char* const src);
}
