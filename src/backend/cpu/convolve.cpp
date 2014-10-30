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

using af::dim4;

namespace cpu
{

template<typename T, typename accType, bool expand>
void one2one_1d(T *optr, T const *iptr, T const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &sStrides)
{
    dim_type start = (expand ? 0 : fDims[0]/2ll);
    dim_type end   = (expand ? oDims[0] : start + sDims[0]);
    for(dim_type i=start; i<end; ++i) {
        accType accum = 0.0;
        for(dim_type f=0; f<fDims[0]; ++f) {
            dim_type iIdx = i-f;
            T s_val = ((iIdx>=0 &&iIdx<sDims[0])? iptr[iIdx*sStrides[0]] : T(0));
            accum += accType(s_val * fptr[f]);
        }
        optr[i-start] = T(accum);
    }
}

template<typename T, typename accType, bool expand>
void one2one_2d(T *optr, T const *iptr, T const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &oStrides,
                dim4 const &sStrides, dim4 const &fStrides)
{
    dim_type jStart = (expand ? 0 : fDims[1]/2ll);
    dim_type jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_type iStart = (expand ? 0 : fDims[0]/2ll);
    dim_type iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_type j=jStart; j<jEnd; ++j) {
        dim_type joff = (j-jStart)*oStrides[1];

        for(dim_type i=iStart; i<iEnd; ++i) {

            accType accum = accType(0);
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

                    accum += accType(s_val * fptr[w_joff+wi*fStrides[0]]);
                }
            }
            optr[joff+i-iStart] = T(accum);
        }
    }
}

template<typename T, typename accType, bool expand>
void one2one_3d(T *optr, T const *iptr, T const *fptr, dim4 const &oDims,
                dim4 const &sDims, dim4 const &fDims, dim4 const &oStrides,
                dim4 const &sStrides, dim4 const &fStrides)
{
    dim_type kStart = (expand ? 0 : fDims[2]/2ll);
    dim_type kEnd   = (expand ? oDims[2] : kStart + sDims[2]);
    dim_type jStart = (expand ? 0 : fDims[1]/2ll);
    dim_type jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_type iStart = (expand ? 0 : fDims[0]/2ll);
    dim_type iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for(dim_type k=kStart; k<kEnd; ++k) {
        dim_type koff = (k-kStart)*oStrides[2];

        for(dim_type j=jStart; j<jEnd; ++j) {
            dim_type joff = (j-jStart)*oStrides[1];

            for(dim_type i=iStart; i<iEnd; ++i) {

                accType accum = accType(0);
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

                            accum += accType(s_val * fptr[w_koff+w_joff+wi*fStrides[0]]);
                        }
                    }
                }
                optr[koff+joff+i-iStart] = T(accum);
            } //i loop ends here
        } // j loop ends here
    } // k loop ends here
}

template<typename T, typename accType, dim_type baseDim, bool expand>
void convolve_nd(T *optr, T const *iptr, T const *fptr,
                dim4 const &oDims, dim4 const &sDims, dim4 const &fDims,
                dim4 const &oStrides, dim4 const &sStrides, dim4 const &fStrides,
                ConvolveBatchKind kind)
{
    T * out       = optr;
    T const *in   = iptr;
    T const *filt = fptr;

    dim_type out_step = 0ll, in_step   = 0ll, filt_step = 0ll;

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
            case 1: one2one_1d<T, accType, expand>(out, in, filt, oDims, sDims, fDims, sStrides);                     break;
            case 2: one2one_2d<T, accType, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
            case 3: one2one_3d<T, accType, expand>(out, in, filt, oDims, sDims, fDims, oStrides, sStrides, fStrides); break;
        }
        out += out_step;
        in  += in_step;
        filt+= filt_step;
    }
}

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> * convolve(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind)
{
    auto sDims    = signal.dims();
    auto fDims    = filter.dims();
    auto sStrides = signal.strides();

    Array<T> *out = nullptr;
    dim4 oDims(1);

    if (expand) {
        for(dim_type d=0; d<4ll; ++d) {
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
    out = createEmptyArray<T>(oDims);

    convolve_nd<T, accT, baseDim, expand>(out->get(), signal.get(), filter.get(),
            oDims, sDims, fDims, out->strides(), sStrides, filter.strides(), kind);

    return out;
}

template<typename T, typename accType, dim_type conv_dim, bool expand>
void convolve2_separable(T *optr, T const *iptr, T const *fptr,
                        dim4 const &oDims, dim4 const &sDims, dim4 const &orgDims, dim_type fDim,
                        dim4 const &oStrides, dim4 const &sStrides, dim_type fStride)
{
    const dim_type outr_dim = (conv_dim+1ll)%2ll;
    dim_type iStart = (expand ? 0 : fDim/2ll);
    dim_type iEnd   = (expand ? oDims[conv_dim] : (iStart + std::min(sDims[conv_dim], orgDims[conv_dim])));
    dim_type jStart = (expand ? 0 : fDim/2ll);
    dim_type jEnd   = (expand ? oDims[outr_dim] : (jStart + std::min(sDims[outr_dim], orgDims[outr_dim])));

    for(dim_type j=jStart; j<jEnd; ++j) {
        dim_type joff = (j-jStart)*oStrides[outr_dim];

        for(dim_type i=iStart; i<iEnd; ++i) {
            dim_type ioff = (i-iStart)*oStrides[conv_dim];

            accType accum = accType(0);

            dim_type s_joff = j * sStrides[outr_dim];
            bool isJValid = (j>=0 && j<sDims[outr_dim]);

            for(dim_type w=0; w<fDim; ++w) {
                // offset right index: we have already
                // choosen the changing index to be pointed
                // by i, therefore no need to check that here
                dim_type idx = i-w;

                dim_type off = idx * sStrides[conv_dim];

                T s_val = T(0);
                if (isJValid && (idx>=0 && idx<sDims[conv_dim])) {
                    // we have already offseted the convolving
                    // dimension, we just need to offset the j offset
                    // to reach corresponding 2d element
                    s_val = iptr[s_joff+off];
                }

                accum += accType(s_val * fptr[w*fStride]);
            }
            optr[joff+ioff] = T(accum);
        }
    }
}

template<typename T, typename accT, bool expand>
Array<T> * convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter)
{
    auto sDims    = signal.dims();
    auto cfDims   = c_filter.dims();
    auto rfDims   = r_filter.dims();
    auto sStrides = signal.strides();

    Array<T> *out = nullptr;

    dim4 tDims(sDims[0]+cfDims[0]-1, sDims[1]+rfDims[0]-1, sDims[2],sDims[3]);
    dim4 oDims(1);

    if (expand) {
        // separable convolve only does ONE2ONE and standard batch(MANY2ONE)
        oDims[0] = sDims[0]+cfDims[0]-1;
        oDims[1] = sDims[1]+rfDims[0]-1;
        oDims[2] = sDims[2];
    } else {
        oDims = sDims;
    }

    Array<T> *temp= createEmptyArray<T>(tDims);
    out           = createEmptyArray<T>(oDims);
    auto tStrides = temp->strides();
    auto oStrides = out->strides();


    for (dim_type b=0; b<oDims[2]; ++b) {
        T const *iptr = signal.get()+ b*sStrides[2];
        T *tptr = temp->get() + b*tStrides[2];
        T *optr = out->get()  + b*oStrides[2];

        convolve2_separable<T, accT, 0, true>(tptr, iptr, c_filter.get(),
                tDims, sDims, sDims, cfDims[0],
                tStrides, sStrides, c_filter.strides()[0]);

        convolve2_separable<T, accT, 1, expand>(optr, tptr, r_filter.get(),
                oDims, tDims, sDims, rfDims[0],
                oStrides, tStrides, r_filter.strides()[0]);
    }

    destroyArray<T>(*temp);

    return out;
}

#define INSTANTIATE(T, accT)  \
    template Array<T> * convolve <T, accT, 1, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 1, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve2<T, accT, true >(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);  \
    template Array<T> * convolve2<T, accT, false>(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}
