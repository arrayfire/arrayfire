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

    template<typename inType, typename outType>
    void copy(Array<outType> &dst, const Array<inType> &src, outType default_value, double factor)
    {
        dim4 src_dims       = src.dims();
        dim4 dst_dims       = dst.dims();
        dim4 src_strides    = src.strides();
        dim4 dst_strides    = dst.strides();

        const inType * src_ptr = src.get();
        outType * dst_ptr      = dst.get();

        dim_type trgt_l = std::min(dst_dims[3], src_dims[3]);
        dim_type trgt_k = std::min(dst_dims[2], src_dims[2]);
        dim_type trgt_j = std::min(dst_dims[1], src_dims[1]);
        dim_type trgt_i = std::min(dst_dims[0], src_dims[0]);

        for(dim_type l=0; l<dst_dims[3]; ++l) {

            dim_type src_loff = l*src_strides[3];
            dim_type dst_loff = l*dst_strides[3];
            bool isLvalid = l<trgt_l;

            for(dim_type k=0; k<dst_dims[2]; ++k) {

                dim_type src_koff = k*src_strides[2];
                dim_type dst_koff = k*dst_strides[2];
                bool isKvalid = k<trgt_k;

                for(dim_type j=0; j<dst_dims[1]; ++j) {

                    dim_type src_joff = j*src_strides[1];
                    dim_type dst_joff = j*dst_strides[1];
                    bool isJvalid = j<trgt_j;

                    for(dim_type i=0; i<dst_dims[0]; ++i) {
                        outType temp = default_value;
                        if (isLvalid && isKvalid && isJvalid && i<trgt_i) {
                            dim_type src_idx = i*src_strides[0] + src_joff + src_koff + src_loff;
                            temp = outType(src_ptr[src_idx])*outType(factor);
                        }
                        dim_type dst_idx = i*dst_strides[0] + dst_joff + dst_koff + dst_loff;
                        dst_ptr[dst_idx] = temp;
                    }
                }
            }
        }
    }

#define INSTANTIATE(T)                                                  \
    template void      copyData<T> (T *data, const Array<T> &from);     \
    template Array<T>* copyArray<T>(const Array<T> &A);                 \

    INSTANTIATE(float  )
    INSTANTIATE(double )
    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(uchar  )
    INSTANTIATE(char   )

#define INSTANTIATE_COPY(SRC_T)                                                       \
    template void copy<SRC_T, float  >(Array<float  > &dst, const Array<SRC_T> &src, float   default_value, double factor); \
    template void copy<SRC_T, double >(Array<double > &dst, const Array<SRC_T> &src, double  default_value, double factor); \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, cfloat  default_value, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, cdouble default_value, double factor); \
    template void copy<SRC_T, int    >(Array<int    > &dst, const Array<SRC_T> &src, int     default_value, double factor); \
    template void copy<SRC_T, uint   >(Array<uint   > &dst, const Array<SRC_T> &src, uint    default_value, double factor); \
    template void copy<SRC_T, uchar  >(Array<uchar  > &dst, const Array<SRC_T> &src, uchar   default_value, double factor); \
    template void copy<SRC_T, char   >(Array<char   > &dst, const Array<SRC_T> &src, char    default_value, double factor);

    INSTANTIATE_COPY(float )
    INSTANTIATE_COPY(double)
    INSTANTIATE_COPY(int   )
    INSTANTIATE_COPY(uint  )
    INSTANTIATE_COPY(uchar )
    INSTANTIATE_COPY(char  )

#define INSTANTIATE_COMPLEX_COPY(SRC_T)                                               \
    template void copy<SRC_T, cfloat >(Array<cfloat > &dst, const Array<SRC_T> &src, cfloat  default_value, double factor); \
    template void copy<SRC_T, cdouble>(Array<cdouble> &dst, const Array<SRC_T> &src, cdouble default_value, double factor);

    INSTANTIATE_COMPLEX_COPY(cfloat )
    INSTANTIATE_COMPLEX_COPY(cdouble)

}
