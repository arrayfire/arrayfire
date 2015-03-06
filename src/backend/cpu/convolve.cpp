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
    dim_type start = (expand ? 0 : fDims[0]/2);
    dim_type end   = (expand ? oDims[0] : start + sDims[0]);
    for(dim_type i=start; i<end; ++i) {
        accT accum = 0.0;
        for(dim_type f=0; f<fDims[0]; ++f) {
            dim_type iIdx = i-f;
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
    dim_type jStart = (expand ? 0 : fDims[1]/2);
    dim_type jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_type iStart = (expand ? 0 : fDims[0]/2);
    dim_type iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_type j=jStart; j<jEnd; ++j) {
        dim_type joff = (j-jStart)*oStrides[1];

        for(dim_type i=iStart; i<iEnd; ++i) {

            accT accum = accT(0);
            for(dim_type wj=0; wj<fDims[1]; ++wj) {
                dim_type jIdx  = j-wj;
                dim_type w_joff = wj*fStrides[1];
                dim_type s_joff = jIdx * sStrides[1];
                bool isJValid = (jIdx>=0 && jIdx<sDims[1]);

                for(dim_type wi=0; wi<fDims[0]; ++wi) {
                    dim_type iIdx = i-wi;

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
    dim_type kStart = (expand ? 0 : fDims[2]/2);
    dim_type kEnd   = (expand ? oDims[2] : kStart + sDims[2]);
    dim_type jStart = (expand ? 0 : fDims[1]/2);
    dim_type jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_type iStart = (expand ? 0 : fDims[0]/2);
    dim_type iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_type k=kStart; k<kEnd; ++k) {
        dim_type koff = (k-kStart)*oStrides[2];

        for(dim_type j=jStart; j<jEnd; ++j) {
            dim_type joff = (j-jStart)*oStrides[1];

            for(dim_type i=iStart; i<iEnd; ++i) {

                accT accum = accT(0);
                for(dim_type wk=0; wk<fDims[2]; ++wk) {
                    dim_type kIdx  = k-wk;
                    dim_type w_koff = wk*fStrides[2];
                    dim_type s_koff = kIdx * sStrides[2];
                    bool isKValid = (kIdx>=0 && kIdx<sDims[2]);

                    for(dim_type wj=0; wj<fDims[1]; ++wj) {
                        dim_type jIdx  = j-wj;
                        dim_type w_joff = wj*fStrides[1];
                        dim_type s_joff = jIdx * sStrides[1];
                        bool isJValid = (jIdx>=0 && jIdx<sDims[1]);

                        for(dim_type wi=0; wi<fDims[0]; ++wi) {
                            dim_type iIdx = i-wi;

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

template<typename T, typename accT, dim_type baseDim, bool expand>
void convolve_nd(T *optr, T const *iptr, accT const *fptr,
                dim4 const &oDims, dim4 const &sDims, dim4 const &fDims,
                dim4 const &oStrides, dim4 const &sStrides, dim4 const &fStrides,
                ConvolveBatchKind kind)
{
    T * out       = optr;
    T const *in   = iptr;
    accT const *filt = fptr;

    dim_type out_step = 0, in_step   = 0, filt_step = 0;

    switch(kind) {
        case MANY2ONE:
            out_step = oStrides[baseDim];
            in_step  = sStrides[baseDim];
            break;
        case MANY2MANY:
            out_step  = oStrides[baseDim];
            in_step   = sStrides[baseDim];
            filt_step = fStrides[baseDim];
            break;
        case ONE2ALL:
            out_step  = oStrides[baseDim];
            filt_step = fStrides[baseDim];
            break;
        default:
            out_step = oStrides[baseDim];
            break;
    }

    dim_type bCount = (kind==ONE2ALL ? fDims[baseDim] : sDims[baseDim]);

    for(dim_type b=0; b<bCount; ++b) {
        switch(baseDim) {
            case 1: one2one_1d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, sStrides);                     break;
            case 2: one2one_2d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
            case 3: one2one_3d<T, accT, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
        }
        out += out_step;
        in  += in_step;
        filt+= filt_step;
    }
}

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind)
{
    auto sDims    = signal.dims();
    auto fDims    = filter.dims();
    auto sStrides = signal.strides();

    dim4 oDims(1);

    if (expand) {
        for(dim_type d=0; d<4; ++d) {
            if (kind==ONE2ONE || kind==ONE2ALL) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==ONE2ALL) oDims[baseDim] = fDims[baseDim];
    }

    Array<T> out = createEmptyArray<T>(oDims);

    convolve_nd<T, accT, baseDim, expand>(out.get(), signal.get(), filter.get(),
            oDims, sDims, fDims, out.strides(), sStrides, filter.strides(), kind);

    return out;
}

template<typename T, typename accT, dim_type conv_dim, bool expand>
void convolve2_separable(T *optr, T const *iptr, accT const *fptr,
                        dim4 const &oDims, dim4 const &sDims, dim4 const &orgDims, dim_type fDim,
                        dim4 const &oStrides, dim4 const &sStrides, dim_type fStride)
{
    for(dim_type j=0; j<oDims[1]; ++j) {

        dim_type jOff = j*oStrides[1];
        dim_type cj = j + (conv_dim==1)*(expand ? 0: fDim>>1);

        for(dim_type i=0; i<oDims[0]; ++i) {

            dim_type iOff = i*oStrides[0];
            dim_type ci = i + (conv_dim==0)*(expand ? 0 : fDim>>1);

            accT accum = scalar<accT>(0);

            for(dim_type f=0; f<fDim; ++f) {
                T f_val = fptr[f];
                T s_val;

                if (conv_dim==0) {
                    dim_type offi = ci - f;
                    bool isCIValid = offi>=0 && offi<sDims[0];
                    bool isCJValid = cj>=0 && cj<sDims[1];
                    s_val = (isCJValid && isCIValid ? iptr[cj*sDims[0]+offi] : scalar<T>(0));
                } else {
                    dim_type offj = cj - f;
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

    dim_type cflen = (dim_type)cfDims.elements();
    dim_type rflen = (dim_type)rfDims.elements();

    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        // separable convolve only does ONE2ONE and standard batch(MANY2ONE)
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);
    auto tStrides = temp.strides();
    auto oStrides = out.strides();

    for (dim_type b=0; b<oDims[2]; ++b) {
        T const *iptr = signal.get()+ b*sStrides[2];
        T *tptr = temp.get() + b*tStrides[2];
        T *optr = out.get()  + b*oStrides[2];

        convolve2_separable<T, accT, 0, expand>(tptr, iptr, c_filter.get(),
                                                tDims, sDims, sDims, cflen,
                                                tStrides, sStrides, c_filter.strides()[0]);

        convolve2_separable<T, accT, 1, expand>(optr, tptr, r_filter.get(),
                                                oDims, tDims, sDims, rflen,
                                                oStrides, tStrides, r_filter.strides()[0]);
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
