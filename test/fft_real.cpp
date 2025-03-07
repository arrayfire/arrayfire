/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::fft;
using af::fft2Norm;
using af::fft3Norm;
using af::fftC2R;
using af::fftNorm;
using af::fftR2C;
using af::randu;
using std::abs;
using std::string;
using std::vector;

template<typename T>
class FFT_REAL : public ::testing::Test {};

typedef ::testing::Types<cfloat, cdouble> TestTypes;
TYPED_TEST_SUITE(FFT_REAL, TestTypes);

template<int rank>
array fft(const array &in, double norm) {
    switch (rank) {
        case 1: return fftNorm(in, norm);
        case 2: return fft2Norm(in, norm);
        case 3: return fft3Norm(in, norm);
        default: return in;
    }
}

#define MY_ASSERT_NEAR(aa, bb, cc) ASSERT_NEAR(abs(aa), abs(bb), (cc))

template<typename Tc, int rank>
void fft_real(dim4 dims) {
    typedef typename dtype_traits<Tc>::base_type Tr;
    SUPPORTED_TYPE_CHECK(Tr);

    dtype ty = (dtype)dtype_traits<Tr>::af_type;
    array a  = randu(dims, ty);

    bool is_odd = dims[0] & 1;

    int dim0 = dims[0] / 2 + 1;

    double norm = 1;
    for (int i = 0; i < rank; i++) norm *= dims[i];
    norm = 1 / norm;

    array as = fftR2C<rank>(a, norm);
    array af = fft<rank>(a, norm);

    vector<Tc> has(as.elements());
    vector<Tc> haf(af.elements());

    as.host(&has[0]);
    af.host(&haf[0]);

    for (int j = 0; j < a.elements() / dims[0]; j++) {
        for (int i = 0; i < dim0; i++) {
            MY_ASSERT_NEAR(haf[j * dims[0] + i], has[j * dim0 + i], 1E-2)
                << "at " << j * dims[0] + i;
        }
    }

    array b = fftC2R<rank>(as, is_odd, 1);

    vector<Tr> ha(a.elements());
    vector<Tr> hb(a.elements());

    a.host(&ha[0]);
    b.host(&hb[0]);

    for (int j = 0; j < a.elements(); j++) { ASSERT_NEAR(ha[j], hb[j], 1E-2); }
}

TYPED_TEST(FFT_REAL, Even1D) { fft_real<TypeParam, 1>(dim4(1024, 256)); }

TYPED_TEST(FFT_REAL, Odd1D) { fft_real<TypeParam, 1>(dim4(625, 256)); }

TYPED_TEST(FFT_REAL, Even2D) { fft_real<TypeParam, 2>(dim4(1024, 256)); }

TYPED_TEST(FFT_REAL, Odd2D) { fft_real<TypeParam, 2>(dim4(625, 256)); }

TYPED_TEST(FFT_REAL, Even3D) { fft_real<TypeParam, 3>(dim4(32, 32, 32)); }

TYPED_TEST(FFT_REAL, Odd3D) { fft_real<TypeParam, 3>(dim4(25, 32, 32)); }
