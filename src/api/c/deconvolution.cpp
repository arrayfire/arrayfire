/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <fft_common.hpp>
#include <fftconvolve.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <reduce.hpp>
#include <select.hpp>
#include <shift.hpp>
#include <unary.hpp>
#include <af/image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

using af::dim4;
using arrayfire::common::cast;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createSubArray;
using detail::createValueArray;
using detail::logicOp;
using detail::padArrayBorders;
using detail::scalar;
using detail::select_scalar;
using detail::shift;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::array;
using std::vector;

const int BASE_DIM = 2;

#if defined(AF_CPU)
// CPU backend uses FFTW or MKL
// FFTW works with any data size, but is optimized for
// size decomposition with prime factors up to
// 13.
const dim_t GREATEST_PRIME_FACTOR = 13;
#else
// cuFFT/clFFT works with any data size, but is optimized
// for size decomposition with prime factors up to
// 7.
const dim_t GREATEST_PRIME_FACTOR = 7;
#endif

template<typename T, typename CT>
Array<T> complexNorm(const Array<CT>& input) {
    auto mag  = detail::abs<T, CT>(input);
    auto TWOS = createValueArray(input.dims(), scalar<T>(2));
    return arithOp<T, af_pow_t>(mag, TWOS, input.dims());
}

std::vector<af_seq> calcPadInfo(dim4& inLPad, dim4& psfLPad, dim4& inUPad,
                                dim4& psfUPad, dim4& odims, dim_t nElems,
                                const dim4& idims, const dim4& fdims) {
    vector<af_seq> index(4);

    for (int d = 0; d < 4; ++d) {
        if (d < BASE_DIM) {
            dim_t pad = idims[d] + fdims[d];

            while (greatestPrimeFactor(pad) > GREATEST_PRIME_FACTOR) { pad++; }

            dim_t diffLen  = pad - idims[d];
            inLPad[d]      = diffLen / 2;
            inUPad[d]      = diffLen / 2 + diffLen % 2;
            psfLPad[d]     = 0;
            psfUPad[d]     = pad - fdims[d];
            odims[d]       = pad;
            index[d].begin = inLPad[d];
            index[d].end   = index[d].begin + idims[d] - 1;
            index[d].step  = 1;

            nElems *= odims[d];
        } else {
            inLPad[d]  = 0;
            psfLPad[d] = 0;
            inUPad[d]  = 0;
            psfUPad[d] = 0;
            odims[d]   = std::max(idims[d], fdims[d]);
            index[d]   = af_span;
        }
    }
    return index;
}

template<typename T, typename CT>
void richardsonLucy(Array<T>& currentEstimate, const Array<T>& in,
                    const Array<CT>& P, const Array<CT>& Pc,
                    const unsigned iters, const float normFactor,
                    const dim4 odims) {
    for (unsigned i = 0; i < iters; ++i) {
        auto fft1  = fft_r2c<CT, T>(currentEstimate, BASE_DIM);
        auto cmul1 = arithOp<CT, af_mul_t>(fft1, P, P.dims());
        auto ifft1 = fft_c2r<CT, T>(cmul1, normFactor, odims, BASE_DIM);
        auto div1  = arithOp<T, af_div_t>(in, ifft1, in.dims());
        auto fft2  = fft_r2c<CT, T>(div1, BASE_DIM);
        auto cmul2 = arithOp<CT, af_mul_t>(fft2, Pc, Pc.dims());
        auto ifft2 = fft_c2r<CT, T>(cmul2, normFactor, odims, BASE_DIM);

        currentEstimate =
            arithOp<T, af_mul_t>(currentEstimate, ifft2, ifft2.dims());
    }
}

template<typename T, typename CT>
void landweber(Array<T>& currentEstimate, const Array<T>& in,
               const Array<CT>& P, const Array<CT>& Pc, const unsigned iters,
               const float relaxFactor, const float normFactor,
               const dim4 odims) {
    const dim4& dims = P.dims();

    auto I        = fft_r2c<CT, T>(in, BASE_DIM);
    auto Pn       = complexNorm<T, CT>(P);
    auto ONE      = createValueArray(dims, scalar<T>(1.0));
    auto alpha    = createValueArray(dims, scalar<T>(relaxFactor));
    auto alphaC   = cast<CT>(alpha);
    auto prod     = arithOp<T, af_mul_t>(alpha, Pn, dims);
    auto lhsFac   = arithOp<T, af_sub_t>(ONE, prod, dims);
    auto lhs      = cast<CT>(lhsFac);
    auto rhsFac   = arithOp<CT, af_mul_t>(Pc, I, dims);
    auto rhs      = arithOp<CT, af_mul_t>(rhsFac, alphaC, dims);
    auto iterTemp = I;

    for (unsigned i = 0; i < iters; ++i) {
        auto mul = arithOp<CT, af_mul_t>(iterTemp, lhs, dims);
        iterTemp = arithOp<CT, af_add_t>(mul, rhs, dims);
    }
    currentEstimate = fft_c2r<CT, T>(iterTemp, normFactor, odims, BASE_DIM);
}

template<typename InputType, typename RealType = float>
af_array iterDeconv(const af_array in, const af_array ker, const uint iters,
                    const float rfactor, const af_iterative_deconv_algo algo) {
    using T    = RealType;
    using CT   = typename std::conditional<std::is_same<T, double>::value,
                                         cdouble, cfloat>::type;
    auto input = castArray<T>(in);
    auto psf   = castArray<T>(ker);
    const dim4& idims = input.dims();
    const dim4& fdims = psf.dims();
    dim_t nElems      = 1;

    dim4 inUPad, psfUPad, inLPad, psfLPad, odims(1);

    auto index = calcPadInfo(inLPad, psfLPad, inUPad, psfUPad, odims, nElems,
                             idims, fdims);
    auto paddedIn =
        padArrayBorders<T>(input, inLPad, inUPad, AF_PAD_CLAMP_TO_EDGE);
    auto paddedPsf = padArrayBorders<T>(psf, psfLPad, psfUPad, AF_PAD_ZERO);

    const std::array<int, 4> shiftDims = {-int(fdims[0] / 2),
                                          -int(fdims[1] / 2), 0, 0};
    auto shiftedPsf                    = shift(paddedPsf, shiftDims.data());

    auto P  = fft_r2c<CT, T>(shiftedPsf, BASE_DIM);
    auto Pc = conj(P);

    Array<T> currentEstimate = paddedIn;
    const double normFactor  = 1 / static_cast<double>(nElems);

    switch (algo) {
        case AF_ITERATIVE_DECONV_RICHARDSONLUCY:
            richardsonLucy(currentEstimate, paddedIn, P, Pc, iters, normFactor,
                           odims);
            break;
        case AF_ITERATIVE_DECONV_LANDWEBER:
        default:
            landweber(currentEstimate, paddedIn, P, Pc, iters, rfactor,
                      normFactor, odims);
    }
    return getHandle(createSubArray<T>(currentEstimate, index));
}

af_err af_iterative_deconv(af_array* out, const af_array in, const af_array ker,
                           const unsigned iterations, const float relax_factor,
                           const af_iterative_deconv_algo algo) {
    try {
        const ArrayInfo& inputInfo  = getInfo(in);
        const dim4& inputDims       = inputInfo.dims();
        const ArrayInfo& kernelInfo = getInfo(ker);
        const dim4& kernelDims      = kernelInfo.dims();

        DIM_ASSERT(2, (inputDims.ndims() == 2));
        DIM_ASSERT(3, (kernelDims.ndims() == 2));
        ARG_ASSERT(4, (iterations > 0));
        ARG_ASSERT(5, std::isfinite(relax_factor));
        ARG_ASSERT(5, (relax_factor > 0));
        ARG_ASSERT(6, (algo == AF_ITERATIVE_DECONV_DEFAULT ||
                       algo == AF_ITERATIVE_DECONV_LANDWEBER ||
                       algo == AF_ITERATIVE_DECONV_RICHARDSONLUCY));
        af_array res   = 0;
        unsigned iters = iterations;
        float rfac     = relax_factor;

        af_dtype inputType = inputInfo.getType();
        switch (inputType) {
            case f32:
                res = iterDeconv<float>(in, ker, iters, rfac, algo);
                break;
            case s16:
                res = iterDeconv<short>(in, ker, iters, rfac, algo);
                break;
            case u16:
                res = iterDeconv<ushort>(in, ker, iters, rfac, algo);
                break;
            case u8: res = iterDeconv<uchar>(in, ker, iters, rfac, algo); break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(res, *out);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename CT>
Array<CT> denominator(const Array<CT>& I, const Array<CT>& P, const float gamma,
                      const af_inverse_deconv_algo algo) {
    using T = typename af::dtype_traits<CT>::base_type;

    auto RCNST = createValueArray(I.dims(), scalar<T>(gamma));

    if (algo == AF_INVERSE_DECONV_TIKHONOV) {
        auto normP = complexNorm<T, CT>(P);
        auto denom = arithOp<T, af_add_t>(normP, RCNST, normP.dims());

        return cast<CT, T>(denom);
    } else {
        // TODO(pradeep) Wiener Filter code path is disabled.
        // This code path doesn't is not exposed using current API
        auto normI = complexNorm<T, CT>(I);
        auto sRes  = arithOp<T, af_sub_t>(normI, RCNST, normI.dims());
        auto dRes  = arithOp<T, af_div_t>(RCNST, sRes, RCNST.dims());
        auto normP = complexNorm<T, CT>(P);
        auto denom = arithOp<T, af_add_t>(normP, dRes, normP.dims());

        return cast<CT, T>(denom);
    }
}

template<typename InputType, typename RealType = float>
af_array invDeconv(const af_array in, const af_array ker, const float gamma,
                   const af_inverse_deconv_algo algo) {
    using T    = RealType;
    using CT   = typename std::conditional<std::is_same<T, double>::value,
                                         cdouble, cfloat>::type;
    auto input = castArray<T>(in);
    auto psf   = castArray<T>(ker);
    const dim4& idims = input.dims();
    const dim4& fdims = psf.dims();
    dim_t nElems      = 1;

    dim4 inUPad, psfUPad, inLPad, psfLPad, odims(1);

    auto index = calcPadInfo(inLPad, psfLPad, inUPad, psfUPad, odims, nElems,
                             idims, fdims);
    auto paddedIn =
        padArrayBorders<T>(input, inLPad, inUPad, AF_PAD_CLAMP_TO_EDGE);
    auto paddedPsf = padArrayBorders<T>(psf, psfLPad, psfUPad, AF_PAD_ZERO);
    const array<int, 4> shiftDims = {-int(fdims[0] / 2), -int(fdims[1] / 2), 0,
                                     0};

    auto shiftedPsf = shift(paddedPsf, shiftDims.data());

    auto I      = fft_r2c<CT, T>(paddedIn, BASE_DIM);
    auto P      = fft_r2c<CT, T>(shiftedPsf, BASE_DIM);
    auto Pc     = conj(P);
    auto numer  = arithOp<CT, af_mul_t>(I, Pc, I.dims());
    auto denom  = denominator(I, P, gamma, algo);
    auto absVal = detail::abs<T, CT>(denom);
    auto THRESH = createValueArray(I.dims(), scalar<T>(gamma));
    auto cond   = logicOp<T, af_ge_t>(absVal, THRESH, absVal.dims());
    auto val    = arithOp<CT, af_div_t>(numer, denom, numer.dims());

    select_scalar<CT, false>(val, cond, val, 0);

    auto ival =
        fft_c2r<CT, T>(val, 1 / static_cast<double>(nElems), odims, BASE_DIM);

    return getHandle(createSubArray<T>(ival, index));
}

af_err af_inverse_deconv(af_array* out, const af_array in, const af_array psf,
                         const float gamma, const af_inverse_deconv_algo algo) {
    try {
        const ArrayInfo& inputInfo = getInfo(in);
        const dim4& inputDims      = inputInfo.dims();
        const ArrayInfo& psfInfo   = getInfo(psf);
        const dim4& psfDims        = psfInfo.dims();

        DIM_ASSERT(2, (inputDims.ndims() == 2));
        DIM_ASSERT(3, (psfDims.ndims() == 2));
        ARG_ASSERT(4, std::isfinite(gamma));
        ARG_ASSERT(4, (gamma > 0));
        ARG_ASSERT(5, (algo == AF_INVERSE_DECONV_DEFAULT ||
                       algo == AF_INVERSE_DECONV_TIKHONOV));
        af_array res = 0;

        af_dtype inputType = inputInfo.getType();
        switch (inputType) {
            case f32: res = invDeconv<float>(in, psf, gamma, algo); break;
            case s16: res = invDeconv<short>(in, psf, gamma, algo); break;
            case u16: res = invDeconv<ushort>(in, psf, gamma, algo); break;
            case u8: res = invDeconv<uchar>(in, psf, gamma, algo); break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(res, *out);
    }
    CATCHALL;
    return AF_SUCCESS;
}
