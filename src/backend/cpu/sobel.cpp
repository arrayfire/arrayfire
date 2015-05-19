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

template<typename Ti, typename To, bool isDX>
void derivative(To *optr, Ti const *iptr, dim4 const &dims, dim4 const &strides)
{
    for(dim_t b3=0; b3<dims[3]; ++b3) {
    for(dim_t b2=0; b2<dims[2]; ++b2) {

        for(dim_t j=0; j<dims[1]; ++j) {

            int joff  = j;
            int _joff = j-1;
            int joff_ = j+1;
            int joffset = j*strides[1];

            for(dim_t i=0; i<dims[0]; ++i) {

                To accum = To(0);

                int  ioff = i;
                int _ioff = i-1;
                int ioff_ = i+1;

                To NW = (_ioff>=0 && _joff>=0) ?
                        iptr[_joff*strides[1]+_ioff*strides[0]] : 0;
                To SW = (ioff_<(int)dims[0] && _joff>=0) ?
                        iptr[_joff*strides[1]+ioff_*strides[0]] : 0;
                To NE = (_ioff>=0 && joff_<(int)dims[1]) ?
                        iptr[joff_*strides[1]+_ioff*strides[0]] : 0;
                To SE = (ioff_<(int)dims[0] && joff_<(int)dims[1]) ?
                        iptr[joff_*strides[1]+ioff_*strides[0]] : 0;

                if (isDX) {
                    To W  = _joff>=0 ?
                            iptr[_joff*strides[1]+ioff*strides[0]] : 0;

                    To E  = joff_<(int)dims[1] ?
                            iptr[joff_*strides[1]+ioff*strides[0]] : 0;

                    accum = NW+SW - (NE+SE) + 2*(W-E);
                } else {
                    To N  = _ioff>=0 ?
                            iptr[joff*strides[1]+_ioff*strides[0]] : 0;

                    To S  = ioff_<(int)dims[0] ?
                            iptr[joff*strides[1]+ioff_*strides[0]] : 0;

                    accum = NW+NE - (SW+SE) + 2*(N-S);
                }

                optr[joffset+i*strides[0]] = accum;
            }
        }

        optr += strides[2];
        iptr += strides[2];
    }
    optr += strides[3];
    iptr += strides[3];
    }
}

template<typename Ti, typename To>
std::pair< Array<To>, Array<To> >
sobelDerivatives(const Array<Ti> &img, const unsigned &ker_size)
{
    Array<To> dx = createEmptyArray<To>(img.dims());
    Array<To> dy = createEmptyArray<To>(img.dims());

    derivative<Ti, To, true >(dx.get(), img.get(), img.dims(), img.strides());
    derivative<Ti, To, false>(dy.get(), img.get(), img.dims(), img.strides());

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(Ti, To)                                                 \
    template std::pair< Array<To>, Array<To> >                            \
    sobelDerivatives(const Array<Ti> &img, const unsigned &ker_size);

INSTANTIATE(float , float)
INSTANTIATE(double, double)
INSTANTIATE(int   , int)
INSTANTIATE(uint  , int)
INSTANTIATE(char  , int)
INSTANTIATE(uchar , int)

}
