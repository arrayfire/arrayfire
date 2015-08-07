/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <type_traits>
#include <af/array.h>
#include <Array.hpp>
#include <copy.hpp>
#include <cstring>
#include <algorithm>
#include <complex>
#include <vector>
#include <cassert>
#include <err_cpu.hpp>
#include <math.hpp>

namespace cpu
{
    template<typename T>
    static void stridedCopy(T* dst, const dim4& ostrides, const T* src, const dim4 &dims, const dim4 &strides, unsigned dim)
    {
        if(dim == 0) {
            if(strides[dim] == 1) {
                //FIXME: Check for errors / exceptions
                memcpy(dst, src, dims[dim] * sizeof(T));
            } else {
                for(dim_t i = 0; i < dims[dim]; i++) {
                    dst[i] = src[strides[dim]*i];
                }
            }
        } else {
            for(dim_t i = dims[dim]; i > 0; i--) {
                stridedCopy<T>(dst, ostrides, src, dims, strides, dim - 1);
                src += strides[dim];
                dst += ostrides[dim];
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
            dim4 ostrides = calcStrides(from.dims());
            stridedCopy<T>(to, ostrides, from.get(), from.dims(), from.strides(), from.ndims() - 1);
        }
    }

    template<typename T>
    Array<T> copyArray(const Array<T> &A)
    {
        Array<T> out = createEmptyArray<T>(A.dims());
        copyData(out.get(), A);
        return out;
    }

    template<typename inType, typename outType>
    static void copy(Array<outType> &dst, const Array<inType> &src, outType default_value, double factor)
    {
        dim4 src_dims       = src.dims();
        dim4 dst_dims       = dst.dims();
        dim4 src_strides    = src.strides();
        dim4 dst_strides    = dst.strides();

        const inType * src_ptr = src.get();
        outType * dst_ptr      = dst.get();

        dim_t trgt_l = std::min(dst_dims[3], src_dims[3]);
        dim_t trgt_k = std::min(dst_dims[2], src_dims[2]);
        dim_t trgt_j = std::min(dst_dims[1], src_dims[1]);
        dim_t trgt_i = std::min(dst_dims[0], src_dims[0]);

        for(dim_t l=0; l<dst_dims[3]; ++l) {

            dim_t src_loff = l*src_strides[3];
            dim_t dst_loff = l*dst_strides[3];
            bool isLvalid = l<trgt_l;

            for(dim_t k=0; k<dst_dims[2]; ++k) {

                dim_t src_koff = k*src_strides[2];
                dim_t dst_koff = k*dst_strides[2];
                bool isKvalid = k<trgt_k;

                for(dim_t j=0; j<dst_dims[1]; ++j) {

                    dim_t src_joff = j*src_strides[1];
                    dim_t dst_joff = j*dst_strides[1];
                    bool isJvalid = j<trgt_j;

                    for(dim_t i=0; i<dst_dims[0]; ++i) {
                        outType temp = default_value;
                        if (isLvalid && isKvalid && isJvalid && i<trgt_i) {
                            dim_t src_idx = i*src_strides[0] + src_joff + src_koff + src_loff;
                            temp = outType(src_ptr[src_idx])*outType(factor);
                        }
                        dim_t dst_idx = i*dst_strides[0] + dst_joff + dst_koff + dst_loff;
                        dst_ptr[dst_idx] = temp;
                    }
                }
            }
        }
    }

    template<typename T>
    void multiply_inplace(Array<T> &in, double val)
    {
        copy<T, T>(in, in, 0, val);
    }

    template<typename inType, typename outType>
    Array<outType>
    padArray(Array<inType> const &in, dim4 const &dims,
             outType default_value, double factor)
    {
        Array<outType> ret = createValueArray<outType>(dims, default_value);
        copy<inType, outType>(ret, in, outType(default_value), factor);
        return ret;
    }

    template<typename inType, typename outType>
    void copyArray(Array<outType> &out, Array<inType> const &in)
    {
        copy<inType, outType>(out, in, scalar<outType>(0), 1.0);
    }


#define INSTANTIATE(T)                                                  \
    template void      copyData<T> (T *data, const Array<T> &from);     \
    template Array<T>  copyArray<T>(const Array<T> &A);                 \
    template void      multiply_inplace<T> (Array<T> &in, double norm); \

    INSTANTIATE(float  )
    INSTANTIATE(double )
    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(uchar  )
    INSTANTIATE(char   )
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )


#define INSTANTIATE_PAD_ARRAY(SRC_T)                                    \
    template Array<float  > padArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value, double factor); \
    template Array<double > padArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value, double factor); \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template Array<int    > padArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value, double factor); \
    template Array<uint   > padArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value, double factor); \
    template Array<intl    > padArray<SRC_T, intl    >(Array<SRC_T> const &src, dim4 const &dims, intl     default_value, double factor); \
    template Array<uintl   > padArray<SRC_T, uintl   >(Array<SRC_T> const &src, dim4 const &dims, uintl    default_value, double factor); \
    template Array<uchar  > padArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value, double factor); \
    template Array<char   > padArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value, double factor); \
    template void copyArray<SRC_T, float  >(Array<float  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, double >(Array<double > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cfloat >(Array<cfloat > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int    >(Array<int    > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uint   >(Array<uint   > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, intl    >(Array<intl    > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uintl   >(Array<uintl   > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, uchar  >(Array<uchar  > &dst, Array<SRC_T> const &src); \
    template void copyArray<SRC_T, char   >(Array<char   > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY(float )
    INSTANTIATE_PAD_ARRAY(double)
    INSTANTIATE_PAD_ARRAY(int   )
    INSTANTIATE_PAD_ARRAY(uint  )
    INSTANTIATE_PAD_ARRAY(intl   )
    INSTANTIATE_PAD_ARRAY(uintl  )
    INSTANTIATE_PAD_ARRAY(uchar )
    INSTANTIATE_PAD_ARRAY(char  )

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                            \
    template Array<cfloat > padArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value, double factor); \
    template Array<cdouble> padArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, double factor); \
    template void copyArray<SRC_T, cfloat  >(Array<cfloat  > &dst, Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, cdouble   >(Array<cdouble > &dst, Array<SRC_T> const &src);

    INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat )
    INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)

#define SPECILIAZE_UNUSED_COPYARRAY(SRC_T, DST_T) \
    template<> void copyArray<SRC_T, DST_T>(Array<DST_T> &out, Array<SRC_T> const &in) \
    {\
        CPU_NOT_SUPPORTED();\
    }

    SPECILIAZE_UNUSED_COPYARRAY(cfloat, double)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, float)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uchar)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, char)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uint)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, int)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, intl)
    SPECILIAZE_UNUSED_COPYARRAY(cfloat, uintl)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, double)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, float)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uchar)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, char)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uint)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, int)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, intl)
    SPECILIAZE_UNUSED_COPYARRAY(cdouble, uintl)

}
