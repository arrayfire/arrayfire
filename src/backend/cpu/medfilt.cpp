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
#include <medfilt.hpp>
#include <err_cpu.hpp>
#include <algorithm>

using af::dim4;

namespace cpu
{

template<typename T, af_border_type pad>
Array<T> medfilt(const Array<T> &in, dim_t w_len, dim_t w_wid)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();
    Array<T> out        = createEmptyArray<T>(dims);
    const dim4 ostrides = out.strides();

    std::vector<T> wind_vals;
    wind_vals.reserve(w_len*w_wid);

    T const * in_ptr = in.get();
    T * out_ptr = out.get();

    for(int b3=0; b3<(int)dims[3]; b3++) {

        for(int b2=0; b2<(int)dims[2]; b2++) {

            for(int col=0; col<(int)dims[1]; col++) {

                int ocol_off = col*ostrides[1];

                for(int row=0; row<(int)dims[0]; row++) {

                    wind_vals.clear();

                    for(int wj=0; wj<(int)w_wid; ++wj) {

                        bool isColOff = false;

                        int im_col = col + wj-w_wid/2;
                        int im_coff;
                        switch(pad) {
                            case AF_PAD_ZERO:
                                im_coff = im_col * istrides[1];
                                if (im_col < 0 || im_col>=(int)dims[1])
                                    isColOff = true;
                                break;
                            case AF_PAD_SYM:
                                {
                                    if (im_col < 0) {
                                        im_col *= -1;
                                        isColOff = true;
                                    }

                                    if (im_col>=(int)dims[1]) {
                                        im_col = 2*((int)dims[1]-1) - im_col;
                                        isColOff = true;
                                    }

                                    im_coff = im_col * istrides[1];
                                }
                                break;
                        }

                        for(int wi=0; wi<(int)w_len; ++wi) {

                            bool isRowOff = false;

                            int im_row = row + wi-w_len/2;
                            int im_roff;
                            switch(pad) {
                                case AF_PAD_ZERO:
                                    im_roff = im_row * istrides[0];
                                    if (im_row < 0 || im_row>=(int)dims[0])
                                        isRowOff = true;
                                    break;
                                case AF_PAD_SYM:
                                    {
                                        if (im_row < 0) {
                                            im_row *= -1;
                                            isRowOff = true;
                                        }

                                        if (im_row>=(int)dims[0]) {
                                            im_row = 2*((int)dims[0]-1) - im_row;
                                            isRowOff = true;
                                        }

                                        im_roff = im_row * istrides[0];
                                    }
                                    break;
                            }

                            if(isRowOff || isColOff) {
                                switch(pad) {
                                    case AF_PAD_ZERO:
                                        wind_vals.push_back(0);
                                        break;
                                    case AF_PAD_SYM:
                                        wind_vals.push_back(in_ptr[im_coff+im_roff]);
                                        break;
                                }
                            } else
                                wind_vals.push_back(in_ptr[im_coff+im_roff]);
                        }
                    }

                    std::stable_sort(wind_vals.begin(),wind_vals.end());
                    int off = wind_vals.size()/2;
                    if (wind_vals.size()%2==0)
                        out_ptr[ocol_off+row*ostrides[0]] = (wind_vals[off]+wind_vals[off-1])/2;
                    else {
                        out_ptr[ocol_off+row*ostrides[0]] = wind_vals[off];
                    }
                }
            }
            in_ptr  += istrides[2];
            out_ptr += ostrides[2];
        }
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> medfilt<T, AF_PAD_ZERO     >(const Array<T> &in, dim_t w_len, dim_t w_wid); \
    template Array<T> medfilt<T, AF_PAD_SYM>(const Array<T> &in, dim_t w_len, dim_t w_wid);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
