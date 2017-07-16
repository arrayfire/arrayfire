/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <cassert>

namespace cpu
{
namespace kernel
{

template<typename Ti, typename To, bool isDX>
void derivative(Param<To> output, CParam<Ti> input)
{
    const af::dim4 dims    = input.dims();
    const af::dim4 istrides = input.strides();
    const af::dim4 ostrides = output.strides();

    for(dim_t b3=0; b3<dims[3]; ++b3) {
        To* optr     = output.get() + b3 * ostrides[3];
        const Ti* iptr     = input.get() + b3 * istrides[3];
        for(dim_t b2=0; b2<dims[2]; ++b2) {

            for(dim_t j=0; j<dims[1]; ++j) {

                int joff  = j;
                int _joff = j-1;
                int joff_ = j+1;
                int joffset = j*ostrides[1];

                for(dim_t i=0; i<dims[0]; ++i) {

                    To accum = To(0);

                    int  ioff = i;
                    int _ioff = i-1;
                    int ioff_ = i+1;

                    To NW = (_ioff>=0 && _joff>=0) ?
                        iptr[_joff*istrides[1]+_ioff*istrides[0]] : 0;
                    To SW = (ioff_<(int)dims[0] && _joff>=0) ?
                        iptr[_joff*istrides[1]+ioff_*istrides[0]] : 0;
                    To NE = (_ioff>=0 && joff_<(int)dims[1]) ?
                        iptr[joff_*istrides[1]+_ioff*istrides[0]] : 0;
                    To SE = (ioff_<(int)dims[0] && joff_<(int)dims[1]) ?
                                   iptr[joff_*istrides[1]+ioff_*istrides[0]] : 0;

                    if (isDX) {
                        To W  = _joff>=0 ?
                                   iptr[_joff*istrides[1]+ioff*istrides[0]] : 0;

                        To E  = joff_<(int)dims[1] ?
                                      iptr[joff_*istrides[1]+ioff*istrides[0]] : 0;

                        accum = NW+SW - (NE+SE) + 2*(W-E);
                    } else {
                        To N  = _ioff>=0 ?
                                   iptr[joff*istrides[1]+_ioff*istrides[0]] : 0;

                        To S  = ioff_<(int)dims[0] ?
                                      iptr[joff*istrides[1]+ioff_*istrides[0]] : 0;

                        accum = NW+NE - (SW+SE) + 2*(N-S);
                    }

                    optr[joffset+i*ostrides[0]] = accum;
                }
            }

            optr += ostrides[2];
            iptr += istrides[2];
        }
    }
}

}
}
