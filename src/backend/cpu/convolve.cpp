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
#include <convolve.hpp>
#include <err_cpu.hpp>
#include <math.hpp>

using af::dim4;

namespace cpu
{

template<typename T, typename accT, bool expand>
void one2one_1d(T *optr, T const *iptr, accT const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &sStrides)
{
    dim_t start = (expand ? 0 : fDims[0]/2);
    dim_t end   = (expand ? oDims[0] : start + sDims[0]);
    for(dim_t i=start; i<end; ++i) {
        accT accum = 0.0;
        for(dim_t f=0; f<fDims[0]; ++f) {
            dim_t iIdx = i-f;
            T s_val = ((iIdx>=0 &&iIdx<sDims[0])? iptr[iIdx*sStrides[0]] : T(0));
            accum += accT(s_val * fptr[f]);
        }
        optr[i-start] = T(accum);
    }
}

template<typename T, typename accT, bool expand>
void one2one_2d(T *optr, T const *iptr, accT const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &oStrides,
                dim4 const &sStrides, dim4 const &fStrides)
{
    dim_t jStart = (expand ? 0 : fDims[1]/2);
    dim_t jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_t iStart = (expand ? 0 : fDims[0]/2);
    dim_t iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_t j=jStart; j<jEnd; ++j) {
        dim_t joff = (j-jStart)*oStrides[1];

        for(dim_t i=iStart; i<iEnd; ++i) {

            accT accum = accT(0);
            for(dim_t wj=0; wj<fDims[1]; ++wj) {
                dim_t jIdx  = j-wj;
                dim_t w_joff = wj*fStrides[1];
                dim_t s_joff = jIdx * sStrides[1];
                bool isJValid = (jIdx>=0 && jIdx<sDims[1]);

                for(dim_t wi=0; wi<fDims[0]; ++wi) {
                    dim_t iIdx = i-wi;

                    T s_val = T(0);
                    if ( isJValid && (iIdx>=0 && iIdx<sDims[0])) {
                        s_val = iptr[s_joff+iIdx*sStrides[0]];
                    }

                    accum += accT(s_val * fptr[w_joff+wi*fStrides[0]]);
                }
            }
            optr[joff+i-iStart] = T(accum);
        }
    }
}

template<typename T, typename accT, bool expand>
void one2one_3d(T *optr, T const *iptr, accT const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &oStrides,
                dim4 const &sStrides, dim4 const &fStrides)
{
    dim_t kStart = (expand ? 0 : fDims[2]/2);
    dim_t kEnd   = (expand ? oDims[2] : kStart + sDims[2]);
    dim_t jStart = (expand ? 0 : fDims[1]/2);
    dim_t jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_t iStart = (expand ? 0 : fDims[0]/2);
    dim_t iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_t k=kStart; k<kEnd; ++k) {
        dim_t koff = (k-kStart)*oStrides[2];

        for(dim_t j=jStart; j<jEnd; ++j) {
            dim_t joff = (j-jStart)*oStrides[1];

            for(dim_t i=iStart; i<iEnd; ++i) {

                accT accum = accT(0);
                for(dim_t wk=0; wk<fDims[2]; ++wk) {
                    dim_t kIdx  = k-wk;
                    dim_t w_koff = wk*fStrides[2];
                    dim_t s_koff = kIdx * sStrides[2];
                    bool isKValid = (kIdx>=0 && kIdx<sDims[2]);

                    for(dim_t wj=0; wj<fDims[1]; ++wj) {
                        dim_t jIdx  = j-wj;
                        dim_t w_joff = wj*fStrides[1];
                        dim_t s_joff = jIdx * sStrides[1];
                        bool isJValid = (jIdx>=0 && jIdx<sDims[1]);

                        for(dim_t wi=0; wi<fDims[0]; ++wi) {
                            dim_t iIdx = i-wi;

                            T s_val = T(0);
                            if ( isKValid && isJValid && (iIdx>=0 && iIdx<sDims[0])) {
                                s_val = iptr[s_koff+s_joff+iIdx*sStrides[0]];
                            }

                            accum += accT(s_val * fptr[w_koff+w_joff+wi*fStrides[0]]);
                        }
                    }
                }
                optr[koff+joff+i-iStart] = T(accum);
            } //i loop ends here
        } // j loop ends here
    } // k loop ends here
}

template<typename T, typename accT, dim_t baseDim, bool expand>
void convolve_nd(T *optr, T const *iptr, accT const *fptr,
                dim4 const &oDims, dim4 const &sDims, dim4 const &fDims,
                dim4 const &oStrides, dim4 const &sStrides, dim4 const &fStrides,
                ConvolveBatchKind kind)
{
    dim_t out_step[4]  = {0, 0, 0, 0}; /* first value is never used, and declared for code simplicity */
    dim_t in_step[4]   = {0, 0, 0, 0}; /* first value is never used, and declared for code simplicity */
    dim_t filt_step[4] = {0, 0, 0, 0}; /* first value is never used, and declared for code simplicity */
    dim_t batch[4]     = {0, 1, 1, 1}; /* first value is never used, and declared for code simplicity */

    for (dim_t i=1; i<4; ++i) {
        switch(kind) {
            case CONVOLVE_BATCH_SIGNAL:
                out_step[i] = oStrides[i];
                in_step[i]  = sStrides[i];
                if (i>=baseDim) batch[i] = sDims[i];
                break;
            case CONVOLVE_BATCH_SAME:
                out_step[i]  = oStrides[i];
                in_step[i]   = sStrides[i];
                filt_step[i] = fStrides[i];
                if (i>=baseDim) batch[i] = sDims[i];
                break;
            case CONVOLVE_BATCH_KERNEL:
                out_step[i]  = oStrides[i];
                filt_step[i] = fStrides[i];
                if (i>=baseDim) batch[i] = fDims[i];
                break;
            default:
                break;
        }
    }

    for (dim_t b3=0; b3<batch[3]; ++b3) {
        for (dim_t b2=0; b2<batch[2]; ++b2) {
            for (dim_t b1=0; b1<batch[1]; ++b1) {

                T * out          = optr + b1 * out_step[1] + b2 * out_step[2] + b3 * out_step[3];
                T const *in      = iptr + b1 *  in_step[1] + b2 *  in_step[2] + b3 *  in_step[3];
                accT const *filt = fptr + b1 *filt_step[1] + b2 *filt_step[2] + b3 *filt_step[3];

                switch(baseDim) {
                    case 1: one2one_1d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, sStrides);                     break;
                    case 2: one2one_2d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
                    case 3: one2one_3d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
                }
            }
        }
    }
}

template<typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind)
{
    auto sDims    = signal.dims();
    auto fDims    = filter.dims();
    auto sStrides = signal.strides();

    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==CONVOLVE_BATCH_NONE || kind==CONVOLVE_BATCH_KERNEL) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==CONVOLVE_BATCH_KERNEL) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    convolve_nd<T, accT, baseDim, expand>(out.get(), signal.get(), filter.get(),
            oDims, sDims, fDims, out.strides(), sStrides, filter.strides(), kind);

    return out;
}

template<typename T, typename accT, dim_t conv_dim, bool expand>
void convolve2_separable(T *optr, T const *iptr, accT const *fptr,
                        dim4 const &oDims, dim4 const &sDims, dim4 const &orgDims, dim_t fDim,
                        dim4 const &oStrides, dim4 const &sStrides, dim_t fStride)
{
    for(dim_t j=0; j<oDims[1]; ++j) {

        dim_t jOff = j*oStrides[1];
        dim_t cj = j + (conv_dim==1)*(expand ? 0: fDim>>1);

        for(dim_t i=0; i<oDims[0]; ++i) {

            dim_t iOff = i*oStrides[0];
            dim_t ci = i + (conv_dim==0)*(expand ? 0 : fDim>>1);

            accT accum = scalar<accT>(0);

            for(dim_t f=0; f<fDim; ++f) {
                T f_val = fptr[f];
                T s_val;

                if (conv_dim==0) {
                    dim_t offi = ci - f;
                    bool isCIValid = offi>=0 && offi<sDims[0];
                    bool isCJValid = cj>=0 && cj<sDims[1];
                    s_val = (isCJValid && isCIValid ? iptr[cj*sDims[0]+offi] : scalar<T>(0));
                } else {
                    dim_t offj = cj - f;
                    bool isCIValid = ci>=0 && ci<sDims[0];
                    bool isCJValid = offj>=0 && offj<sDims[1];
                    s_val = (isCJValid && isCIValid ? iptr[offj*sDims[0]+ci] : scalar<T>(0));
                }

                accum += accT(s_val * f_val);
            }
            optr[iOff+jOff] = T(accum);
        }
    }
}

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter)
{
    auto sDims    = signal.dims();
    auto cfDims   = c_filter.dims();
    auto rfDims   = r_filter.dims();
    auto sStrides = signal.strides();

    dim_t cflen = (dim_t)cfDims.elements();
    dim_t rflen = (dim_t)rfDims.elements();

    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        // separable convolve only does CONVOLVE_BATCH_NONE and standard batch(CONVOLVE_BATCH_SIGNAL)
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);
    auto tStrides = temp.strides();
    auto oStrides = out.strides();

    for (dim_t b3=0; b3<oDims[3]; ++b3) {

        dim_t i_b3Off = b3*sStrides[3];
        dim_t t_b3Off = b3*tStrides[3];
        dim_t o_b3Off = b3*oStrides[3];

        for (dim_t b2=0; b2<oDims[2]; ++b2) {

            T const *iptr = signal.get()+ b2*sStrides[2] + i_b3Off;
            T *tptr = temp.get() + b2*tStrides[2] + t_b3Off;
            T *optr = out.get()  + b2*oStrides[2] + o_b3Off;

            convolve2_separable<T, accT, 0, expand>(tptr, iptr, c_filter.get(),
                    tDims, sDims, sDims, cflen,
                    tStrides, sStrides, c_filter.strides()[0]);

            convolve2_separable<T, accT, 1, expand>(optr, tptr, r_filter.get(),
                    oDims, tDims, sDims, rflen,
                    oStrides, tStrides, r_filter.strides()[0]);
        }
    }

    return out;
}

#define INSTANTIATE(T, accT)                                            \
    template Array<T> convolve <T, accT, 1, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 1, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve2<T, accT, true >(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter); \
    template Array<T> convolve2<T, accT, false>(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}
