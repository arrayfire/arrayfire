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
    static void stridedCopy(T* dst, const T* src, const dim4 &dims, const dim4 &strides, unsigned dim)
    {
        if(dim == 0) {
            if(strides[dim] == 1) {
                //FIXME: Check for errors / exceptions
                memcpy(dst, src, dims[dim] * sizeof(T));
            } else {
                for(dim_type i = 0; i < dims[dim]; i++) {
                    dst[i] = src[strides[dim]*i];
                }
            }
        } else {
            for(dim_type i = dims[dim]; i > 0; i--) {
                stridedCopy<T>(dst, src, dims, strides, dim - 1);
                src += strides[dim];
                dst += dims[dim-1];
            }
        }
    }

    // Assigns to single elements
    template<typename T>
    void copyData(T *to, const Array<T> &from)
    {
        if(from.isOwner()) {
            // FIXME: Check for errors / exceptions
            memcpy(to, from.get(), from.elements()*sizeof(T));
        } else {
            stridedCopy<T>(to, from.get(), from.dims(), from.strides(), from.ndims() - 1);
        }
    }


    template<typename T>
    Array<T> *copyArray(const Array<T> &A)
    {
        Array<T> *out = createEmptyArray<T>(A.dims());
        copyData(out->get(), A);
        return out;
    }


#define INSTANTIATE(T)                                                  \
    template void      copyData<T> (T *data, const Array<T> &from);     \
    template Array<T>* copyArray<T>(const Array<T> &A);                 \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
