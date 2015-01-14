/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <sobel.hpp>
#include <convolve.hpp>
#include <err_cpu.hpp>
#include <utility>

using af::dim4;

namespace cpu
{

template<typename T, typename accType, bool isDX>
void derivative(T *optr, T const *iptr, dim4 const &dims, dim4 const &strides)
{
    dim_type bCount = dims[2]*dims[3];

    for(dim_type b=0; b<bCount; ++b) {

        for(dim_type j=0; j<dims[1]; ++j) {

            dim_type joff  = j;
            dim_type _joff = j-1;
            dim_type joff_ = j+1;
            dim_type joffset = j*strides[1];

            for(dim_type i=0; i<dims[0]; ++i) {

                accType accum = accType(0);

                dim_type  ioff = i;
                dim_type _ioff = i-1;
                dim_type ioff_ = i+1;

                accType NW = (_ioff>=0 && _joff>=0) ?
                    iptr[_joff*strides[1]+_ioff*strides[0]] : 0;
                accType SW = (ioff_<dims[0] && _joff>=0) ?
                    iptr[_joff*strides[1]+ioff_*strides[0]] : 0;
                accType NE = (_ioff>=0 && joff_<dims[1]) ?
                    iptr[joff_*strides[1]+_ioff*strides[0]] : 0;
                accType SE = (ioff_<dims[0] && joff_<dims[1]) ?
                    iptr[joff_*strides[1]+ioff_*strides[0]] : 0;

                if (isDX) {
                    accType W  = _joff>=0 ?
                        iptr[_joff*strides[1]+ioff*strides[0]] : 0;

                    accType E  = joff_<dims[1] ?
                        iptr[joff_*strides[1]+ioff*strides[0]] : 0;

                    accum = NW+SW - (NE+SE) + 2*(W-E);
                } else {
                    accType N  = _ioff>=0 ?
                        iptr[joff*strides[1]+_ioff*strides[0]] : 0;

                    accType S  = ioff_<dims[0] ?
                        iptr[joff*strides[1]+ioff_*strides[0]] : 0;

                    accum = NW+NE - (SW+SE) + 2*(N-S);
                }

                optr[joffset+i*strides[0]] = T(accum);
            }
        }

        optr += strides[2];
        iptr += strides[2];
    }
}

template<typename T>
std::pair< Array<T>*, Array<T>* >
sobelDerivatives(const Array<T> &img, const unsigned &ker_size)
{
    Array<T> *dx = createEmptyArray<T>(img.dims());
    Array<T> *dy = createEmptyArray<T>(img.dims());

    if (std::is_same<T, double>::value) {
        derivative<T, double, true >(dx->get(), img.get(), img.dims(), img.strides());
        derivative<T, double, false>(dy->get(), img.get(), img.dims(), img.strides());
    }
    else {
        derivative<T, float, true >(dx->get(), img.get(), img.dims(), img.strides());
        derivative<T, float, false>(dy->get(), img.get(), img.dims(), img.strides());
    }

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(T)\
    template std::pair< Array<T>*, Array<T>* > sobelDerivatives(const Array<T> &img, const unsigned &ker_size);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )

}
