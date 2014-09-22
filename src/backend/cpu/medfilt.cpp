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

template<typename T, af_pad_type pad>
Array<T> * medfilt(const Array<T> &in, dim_type w_len, dim_type w_wid)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();

    Array<T> * out      = createEmptyArray<T>(dims);

    const dim4 ostrides = out->strides();

    std::vector<T> wind_vals;
    wind_vals.reserve(w_len*w_wid);

    for(dim_type batchId=0; batchId<dims[2]; batchId++) {

        T const * in_ptr = in.get() + batchId*istrides[2];
        T * out_ptr = out->get() + batchId*ostrides[2];

        for(dim_type col=0; col<dims[1]; col++) {

            dim_type ocol_off = col*ostrides[1];

            for(dim_type row=0; row<dims[0]; row++) {

                wind_vals.clear();

                for(dim_type wj=0; wj<w_wid; ++wj) {

                    bool isColOff = false;

                    dim_type im_col = col + wj-w_wid/2;
                    dim_type im_coff;
                    switch(pad) {
                        case AF_ZERO:
                            im_coff = im_col * istrides[1];
                            if (im_col < 0 || im_col>=dims[1])
                                isColOff = true;
                            break;
                        case AF_SYMMETRIC:
                            {
                                if (im_col < 0) {
                                    im_col *= -1;
                                    isColOff = true;
                                }

                                if (im_col>=dims[1]) {
                                    im_col = 2*(dims[1]-1) - im_col;
                                    isColOff = true;
                                }

                                im_coff = im_col * istrides[1];
                            }
                            break;
                    }

                    for(dim_type wi=0; wi<w_len; ++wi) {

                        bool isRowOff = false;

                        dim_type im_row = row + wi-w_len/2;
                        dim_type im_roff;
                        switch(pad) {
                            case AF_ZERO:
                                im_roff = im_row * istrides[0];
                                if (im_row < 0 || im_row>=dims[0])
                                    isRowOff = true;
                                break;
                            case AF_SYMMETRIC:
                                {
                                    if (im_row < 0) {
                                        im_row *= -1;
                                        isRowOff = true;
                                    }

                                    if (im_row>=dims[0]) {
                                        im_row = 2*(dims[0]-1) - im_row;
                                        isRowOff = true;
                                    }

                                    im_roff = im_row * istrides[0];
                                }
                                break;
                        }

                        if(isRowOff || isColOff) {
                            switch(pad) {
                                case AF_ZERO:
                                    wind_vals.push_back(0);
                                    break;
                                case AF_SYMMETRIC:
                                    wind_vals.push_back(in_ptr[im_coff+im_roff]);
                                    break;
                            }
                        } else
                            wind_vals.push_back(in_ptr[im_coff+im_roff]);
                    }
                }

                std::stable_sort(wind_vals.begin(),wind_vals.end());
                dim_type off = wind_vals.size()/2;
                if (wind_vals.size()%2==0)
                    out_ptr[ocol_off+row*ostrides[0]] = (wind_vals[off]+wind_vals[off-1])/2;
                else {
                    out_ptr[ocol_off+row*ostrides[0]] = wind_vals[off];
                }
            }
        }
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * medfilt<T, AF_ZERO     >(const Array<T> &in, dim_type w_len, dim_type w_wid); \
    template Array<T> * medfilt<T, AF_SYMMETRIC>(const Array<T> &in, dim_type w_len, dim_type w_wid);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
