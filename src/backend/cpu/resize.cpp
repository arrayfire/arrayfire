/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <resize.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <math.hpp>
#include <types.hpp>
#include <af/traits.hpp>

namespace cpu
{
    /**
     * noop function for round to avoid compilation
     * issues due to lack of this function in C90 based
     * compilers, it is only present in C99 and C++11
     *
     * This is not a full fledged implementation, this function
     * is to be used only for positive numbers, i m using it here
     * for calculating dimensions of arrays
     */
    dim_t round2int(float value)
    {
        return (dim_t)(value+0.5f);
    }

    using std::conditional;
    using std::is_same;

    template<typename T>
    using wtype_t = typename conditional<is_same<T, double>::value, double, float>::type;

    template<typename T>
    using vtype_t = typename conditional<is_complex<T>::value,
                                         T, wtype_t<T>
                                        >::type;

    template<typename T, af_interp_type method>
    struct resize_op
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const dim_t x, const dim_t y)
        {
            return;
        }
    };

    template<typename T>
    struct resize_op<T, AF_INTERP_NEAREST>
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                const af::dim4 &ostrides, const af::dim4 &istrides,
                const dim_t x, const dim_t y)
        {
            // Compute Indices
            dim_t i_x = round2int((float)x / (odims[0] / (float)idims[0]));
            dim_t i_y = round2int((float)y / (odims[1] / (float)idims[1]));

            if (i_x >= idims[0]) i_x = idims[0] - 1;
            if (i_y >= idims[1]) i_y = idims[1] - 1;

            dim_t i_off = i_y * istrides[1] + i_x;
            dim_t o_off =   y * ostrides[1] + x;
            // Copy values from all channels
            for(dim_t w = 0; w < odims[3]; w++) {
                dim_t wost = w * ostrides[3];
                dim_t wist = w * istrides[3];
                for(dim_t z = 0; z < odims[2]; z++) {
                    outPtr[o_off + z * ostrides[2] + wost] = inPtr[i_off + z * istrides[2] + wist];
                }
            }
        }
    };

    template<typename T>
    struct resize_op<T, AF_INTERP_BILINEAR>
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                const af::dim4 &ostrides, const af::dim4 &istrides,
                const dim_t x, const dim_t y)
        {
            // Compute Indices
            float f_x = (float)x / (odims[0] / (float)idims[0]);
            float f_y = (float)y / (odims[1] / (float)idims[1]);

            dim_t i1_x  = floor(f_x);
            dim_t i1_y  = floor(f_y);

            if (i1_x >= idims[0]) i1_x = idims[0] - 1;
            if (i1_y >= idims[1]) i1_y = idims[1] - 1;

            float b   = f_x - i1_x;
            float a   = f_y - i1_y;

            dim_t i2_x  = (i1_x + 1 >= idims[0] ? idims[0] - 1 : i1_x + 1);
            dim_t i2_y  = (i1_y + 1 >= idims[1] ? idims[1] - 1 : i1_y + 1);

            typedef typename dtype_traits<T>::base_type BT;
            typedef wtype_t<BT> WT;
            typedef vtype_t<T> VT;

            dim_t o_off = y * ostrides[1] + x;
            // Copy values from all channels
            for(dim_t w = 0; w < odims[3]; w++) {
                dim_t wst = w * istrides[3];
                for(dim_t z = 0; z < odims[2]; z++) {
                    dim_t zst = z * istrides[2];
                    dim_t channel_off = zst + wst;
                    VT p1 = inPtr[i1_y * istrides[1] + i1_x + channel_off];
                    VT p2 = inPtr[i2_y * istrides[1] + i1_x + channel_off];
                    VT p3 = inPtr[i1_y * istrides[1] + i2_x + channel_off];
                    VT p4 = inPtr[i2_y * istrides[1] + i2_x + channel_off];

                    outPtr[o_off + z * ostrides[2] + w * ostrides[3]] =
                                    scalar<WT>((1.0f - a) * (1.0f - b)) * p1 +
                                    scalar<WT>((    a   ) * (1.0f - b)) * p2 +
                                    scalar<WT>((1.0f - a) * (    b   )) * p3 +
                                    scalar<WT>((    a   ) * (    b   )) * p4;
                }
            }
        }
    };

    template<typename T, af_interp_type method>
    void resize_(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                 const af::dim4 &ostrides, const af::dim4 &istrides)
    {
        resize_op<T, method> op;
        for(dim_t y = 0; y < odims[1]; y++) {
            for(dim_t x = 0; x < odims[0]; x++) {
                op(outPtr, inPtr, odims, idims, ostrides, istrides, x, y);
            }
        }
    }

    template<typename T>
    Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                    const af_interp_type method)
    {
        af::dim4 idims = in.dims();
        af::dim4 odims(odim0, odim1, idims[2], idims[3]);

        // Create output placeholder
        Array<T> outArray = createValueArray(odims, (T)0);

        // Get pointers to raw data
        const T *inPtr = in.get();
              T *outPtr = outArray.get();

        af::dim4 ostrides = outArray.strides();
        af::dim4 istrides = in.strides();

        switch(method) {
            case AF_INTERP_NEAREST:
                resize_<T, AF_INTERP_NEAREST>(outPtr, inPtr, odims, idims, ostrides, istrides);
                break;
            case AF_INTERP_BILINEAR:
                resize_<T, AF_INTERP_BILINEAR>(outPtr, inPtr, odims, idims, ostrides, istrides);
                break;
            default:
                break;
        }
        return outArray;
    }


#define INSTANTIATE(T)                                                                            \
    template Array<T> resize<T> (const Array<T> &in, const dim_t odim0, const dim_t odim1, \
                                 const af_interp_type method);


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
