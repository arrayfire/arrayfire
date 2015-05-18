/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <diagonal.hpp>
#include <math.hpp>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename T>
    Array<T> diagCreate(const Array<T> &in, const int num)
    {
        int size = in.dims()[0] + std::abs(num);
        int batch = in.dims()[1];
        Array<T> out = createEmptyArray<T>(dim4(size, size, batch));

        const T *iptr = in.get();
        T *optr = out.get();

        for (int k = 0; k < batch; k++) {
            for (int j = 0; j < size; j++) {
                for (int i = 0; i < size; i++) {
                    T val = scalar<T>(0);
                    if (i == j - num) {
                        val = (num > 0) ? iptr[i] : iptr[j];
                    }
                    optr[i + j * out.strides()[1]] = val;
                }
            }
            optr += out.strides()[2];
            iptr += in.strides()[1];
        }

        return out;
    }

    template<typename T>
    Array<T> diagExtract(const Array<T> &in, const int num)
    {
        const dim_t *idims = in.dims().get();
        dim_t size = std::max(idims[0], idims[1]) - std::abs(num);
        Array<T> out = createEmptyArray<T>(dim4(size, 1, idims[2], idims[3]));

        const dim_t *odims = out.dims().get();

        const int i_off = (num > 0) ? (num * in.strides()[1]) : (-num);

        for (int l = 0; l < (int)odims[3]; l++) {

            for (int k = 0; k < (int)odims[2]; k++) {
                const T *iptr = in.get() + l * in.strides()[3] + k * in.strides()[2] + i_off;
                T *optr = out.get() + l * out.strides()[3] + k * out.strides()[2];

                for (int i = 0; i < (int)odims[0]; i++) {
                    T val = scalar<T>(0);
                    if (i < idims[0] && i < idims[1]) val =  iptr[i * in.strides()[1] + i];
                    optr[i] = val;
                }
            }
        }

        return out;
    }

#define INSTANTIATE_DIAGONAL(T)                                          \
    template Array<T>  diagExtract<T>    (const Array<T> &in, const int num); \
    template Array<T>  diagCreate <T>    (const Array<T> &in, const int num);

    INSTANTIATE_DIAGONAL(float)
    INSTANTIATE_DIAGONAL(double)
    INSTANTIATE_DIAGONAL(cfloat)
    INSTANTIATE_DIAGONAL(cdouble)
    INSTANTIATE_DIAGONAL(int)
    INSTANTIATE_DIAGONAL(uint)
    INSTANTIATE_DIAGONAL(intl)
    INSTANTIATE_DIAGONAL(uintl)
    INSTANTIATE_DIAGONAL(char)
    INSTANTIATE_DIAGONAL(uchar)

}
