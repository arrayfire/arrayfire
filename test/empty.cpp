/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

using namespace af;

template<typename T>
class Array : public ::testing::Test {};

TEST(Array, TestEmptyAssignment) {
    array A     = randu(5, f32);
    array C     = constant(0, 0);
    array B     = A(isNaN(A));
    A(isNaN(A)) = C;
    ASSERT_EQ(B.numdims(), 0u);
    ASSERT_EQ(A.numdims(), 1u);
    ASSERT_EQ(lookup(constant(1, 9), constant(0, 0)).numdims(), 0u);
}

TEST(Array, TestEmptySigProc) {
    ASSERT_EQ(convolve(constant(1, 1), constant(0, 0)).numdims(), 1u);
    ASSERT_EQ(convolve(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(convolve2(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(convolve3(constant(0, 0), constant(0, 0)).numdims(), 0u);

    ASSERT_EQ(iir(constant(0, 0), constant(0, 0), constant(0, 0)).numdims(),
              0u);

    ASSERT_EQ(approx1(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(approx1(constant(0, 0), seq(0, 10)).numdims(), 0u);
    ASSERT_EQ(approx2(constant(0, 0), constant(0, 0), constant(0, 0)).numdims(),
              0u);
    ASSERT_EQ(approx2(constant(0, 0), seq(0, 10), seq(0, 10)).numdims(), 0u);
}

TEST(Array, TestEmptySet) {
    ASSERT_EQ(setIntersect(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(setUnique(constant(0, 0)).numdims(), 0u);

    array A = randu(5, f32);
    array B = constant(0, 0);

    ASSERT_EQ(setUnion(A, B).elements(), 5);
    ASSERT_EQ(setUnion(B, A).elements(), 5);
}

TEST(Array, TestEmptyOperators) {
    ASSERT_EQ((constant(0, 0) + constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) && constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) - constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) & constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) | constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) ^ constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) << constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) >> constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) / constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) == constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) <= constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) >= constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) > constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) < constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(-constant(0, 0).numdims(), 0u);
    ASSERT_EQ((!constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) != constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) += 1).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) -= 1).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) *= 1).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) /= 1).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) || constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) % constant(0, 0)).numdims(), 0u);
    ASSERT_EQ((constant(0, 0) * constant(0, 0)).numdims(), 0u);
}

TEST(Array, TestEmptyFFT) {
    array arr = constant(0, 0);
    fftInPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    fft2InPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    fft3InPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    ifftInPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    ifft2InPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    ifft3InPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);

    ASSERT_EQ((fft(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftNorm(constant(0, 0), 0.5)).numdims(), 0u);
    ASSERT_EQ((fft2(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fft2Norm(constant(0, 0), 0.5)).numdims(), 0u);
    ASSERT_EQ((fft3(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fft3Norm(constant(0, 0), 0.5)).numdims(), 0u);
    ASSERT_EQ((fftC2R<1>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftR2C<1>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftC2R<2>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftR2C<2>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftC2R<3>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((fftR2C<3>(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((ifft(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((ifftNorm(constant(0, 0), 0.5)).numdims(), 0u);
    ASSERT_EQ((ifft2(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((ifft2Norm(constant(0, 0), 0.5)).numdims(), 0u);
    ASSERT_EQ((ifft3(constant(0, 0))).numdims(), 0u);
    ASSERT_EQ((ifft3Norm(constant(0, 0), 0.5)).numdims(), 0u);
}

TEST(Array, TestEmptyDiff) {
    ASSERT_EQ(diff1(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(diff1(constant(1, 1)).numdims(), 0u);
    ASSERT_EQ(diff1(constant(1, 2)).numdims(), 1u);
    ASSERT_EQ(diff2(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(diff2(constant(1, 1)).numdims(), 0u);
    ASSERT_EQ(diff2(constant(1, 2)).numdims(), 0u);
    ASSERT_EQ(diff2(constant(1, 3)).numdims(), 1u);
}

TEST(Array, TestEmptyLinAlg) {
    ASSERT_EQ(det<float>(constant(0, 0)), 1);
    ASSERT_EQ(det<cfloat>(constant(0, 0)).real, 1);
    ASSERT_EQ(det<cdouble>(constant(0, 0)).real, 1);
    ASSERT_EQ(norm(constant(0, 0)), 0);
    ASSERT_EQ(rank(constant(0, 0)), 0u);

    array tau_qr, arr = constant(0, 0);
    qrInPlace(tau_qr, arr);
    ASSERT_EQ(tau_qr.numdims(), 0u);

    array out_qr;
    qr(out_qr, tau_qr, constant(0, 0));
    ASSERT_EQ(out_qr.numdims(), 0u);
    ASSERT_EQ(tau_qr.numdims(), 0u);
    ASSERT_EQ(solve(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(solveLU(constant(0, 0), constant(0, 0), constant(0, 0)).numdims(),
              0u);

    array out_lu, piv_lu;
    lu(out_lu, piv_lu, constant(0, 0));
    ASSERT_EQ(out_lu.numdims(), 0u);
    ASSERT_EQ(piv_lu.numdims(), 0u);

    array low_lu, up_lu;
    lu(low_lu, up_lu, piv_lu, constant(0, 0));
    ASSERT_EQ(low_lu.numdims(), 0u);
    ASSERT_EQ(up_lu.numdims(), 0u);
    ASSERT_EQ(piv_lu.numdims(), 0u);

    luInPlace(piv_lu, arr, true);
    ASSERT_EQ(piv_lu.numdims(), 0u);
    ASSERT_EQ(arr.numdims(), 0u);

    array u, s, v;
    svd(u, s, v, constant(0, 0));
    svdInPlace(u, s, v, arr);
    ASSERT_EQ(dot(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(transpose(constant(0, 0)).numdims(), 0u);
    choleskyInPlace(arr);
    ASSERT_EQ(arr.numdims(), 0u);
    array out;
    cholesky(out, constant(0, 0));
    ASSERT_EQ(out.numdims(), 0u);
}

TEST(Array, TestEmptyMath) {
    ASSERT_EQ(acos(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(acosh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(abs(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(asin(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(asinh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(atan(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(atan2(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(atanh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(cos(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(cosh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(log(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(log10(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(log1p(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sin(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sinh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(tan(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(tanh(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sqrt(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(real(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(imag(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(conjg(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(erf(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(erfc(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(exp(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(expm1(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(cbrt(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(ceil(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(lgamma(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(pow(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(root(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(tgamma(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(arg(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(floor(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(hypot(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(rem(constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(round(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sign(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(trunc(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(factorial(constant(0, 0)).numdims(), 0u);
    // ASSERT_EQ(complex(constant(0,0), constant(0,0)).numdims(), 0u);
}

TEST(Array, TestEmptyVecOp) {
    ASSERT_EQ(accum(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(allTrue(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(anyTrue(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(count(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(where(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(max(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(min(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(product(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sum(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(sort(constant(0, 0)).numdims(), 0u);

    array skeys, svals;
    sort(skeys, svals, constant(0, 0), constant(0, 0));
    ASSERT_EQ(skeys.numdims(), 0u);
    ASSERT_EQ(svals.numdims(), 0u);

    array sout, sind;
    sort(sout, sind, constant(0, 0));
    ASSERT_EQ(sout.numdims(), 0u);
    ASSERT_EQ(sind.numdims(), 0u);
}

TEST(Array, TestEmptyArrMod) {
    ASSERT_EQ(diag(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(diag(constant(0, 0), true).numdims(), 0u);
    ASSERT_EQ(identity(0).numdims(), 0u);
    ASSERT_EQ(iota(dim4(0)).numdims(), 0u);
    ASSERT_EQ(lower(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(upper(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(constant(0, 0).as(u8).numdims(), 0u);
    ASSERT_EQ(isNaN(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(isInf(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(iszero(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(flat(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(flip(constant(0, 0), 0).numdims(), 0u);
    ASSERT_EQ(moddims(constant(0, 0), dim4(0)).numdims(), 0u);
    ASSERT_EQ(reorder(constant(0, 0), 0).numdims(), 0u);
    ASSERT_EQ(shift(constant(0, 0), 1).numdims(), 0u);
    ASSERT_EQ(tile(constant(0, 0), 1).numdims(), 0u);

    ASSERT_EQ(join(0, constant(0, 0), constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(join(0, randu(3), constant(0, 0)).elements(), 3);
    ASSERT_EQ(join(0, constant(0, 0), randn(3)).elements(), 3);

    ASSERT_EQ(select(constant(0, 0), constant(0, 0), constant(0, 0)).numdims(),
              0u);

    array arr = constant(0, 0);
    replace(arr, constant(0, 0), constant(0, 0));
    ASSERT_EQ(arr.numdims(), 0u);
}

TEST(Array, TestEmptyImage) {
    ASSERT_EQ(histogram(constant(0, 0), 1).numdims(), 0u);
    ASSERT_EQ(hsv2rgb(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(gray2rgb(constant(0, 0)).numdims(), 0u);
    ASSERT_EQ(rotate(constant(0, 0), 0).numdims(), 0u);

    af_array h, hout;
    dim_t ds[1];
    af_constant(&h, 0, 0, ds, f32);
    af_histogram(&hout, h, 10, 0.0, 1.0);

    unsigned nd;
    af_get_numdims(&nd, h);
    ASSERT_EQ(nd, 0u);
    af_get_numdims(&nd, hout);
    ASSERT_EQ(nd, 0u);
    ASSERT_SUCCESS(af_release_array(h));
    ASSERT_SUCCESS(af_release_array(hout));
}
