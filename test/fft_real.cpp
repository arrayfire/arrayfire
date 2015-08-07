/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <testHelpers.hpp>

using std::string;
using std::vector;
using af::cfloat;
using af::cdouble;


template<typename T>
class FFT_REAL : public ::testing::Test
{
};

typedef ::testing::Types<af::cfloat, af::cdouble> TestTypes;
TYPED_TEST_CASE(FFT_REAL, TestTypes);

template<int rank>
af::array fft(const af::array &in, double norm)
{
    switch(rank) {
    case 1: return af::fftNorm(in, norm);
    case 2: return af::fft2Norm(in, norm);
    case 3: return af::fft3Norm(in, norm);
    default: return in;
    }
}

#define MY_ASSERT_NEAR(aa, bb, cc) ASSERT_NEAR(abs(aa), abs(bb), (cc))

template<typename Tc, int rank>
void fft_real(af::dim4 dims)
{
    typedef typename af::dtype_traits<Tc>::base_type Tr;
    if (noDoubleTests<Tr>()) return;

    af::dtype ty = (af::dtype)af::dtype_traits<Tr>::af_type;
    af::array a = af::randu(dims, ty);

    bool is_odd = dims[0] & 1;

    int dim0 = dims[0] / 2 + 1;

    double norm = 1;
    for (int i = 0; i < rank; i++) norm *= dims[i];
    norm = 1/norm;

    af::array as = af::fftR2C<rank>(a, norm);
    af::array af = fft<rank>(a, norm);


    std::vector<Tc> has(as.elements());
    std::vector<Tc> haf(af.elements());

    as.host(&has[0]);
    af.host(&haf[0]);

    for (int j = 0; j < a.elements() / dims[0]; j++) {
        for (int i = 0; i < dim0; i++) {
            MY_ASSERT_NEAR(haf[j * dims[0] + i], has[j * dim0 + i], 1E-2) << "at " << j * dims[0] + i;
        }
    }

    af::array b = af::fftC2R<rank>(as, is_odd, 1);

    std::vector<Tr> ha(a.elements());
    std::vector<Tr> hb(a.elements());

    a.host(&ha[0]);
    b.host(&hb[0]);

    for (int j = 0; j < a.elements(); j++) {
        ASSERT_NEAR(ha[j], hb[j], 1E-2);
    }
}

TYPED_TEST(FFT_REAL, Even1D)
{
    fft_real<TypeParam, 1>(af::dim4(1024, 256));
}

TYPED_TEST(FFT_REAL, Odd1D)
{
    fft_real<TypeParam, 1>(af::dim4(625, 256));
}

TYPED_TEST(FFT_REAL, Even2D)
{
    fft_real<TypeParam, 2>(af::dim4(1024, 256));
}

TYPED_TEST(FFT_REAL, Odd2D)
{
    fft_real<TypeParam, 2>(af::dim4(625, 256));
}

TYPED_TEST(FFT_REAL, Even3D)
{
    fft_real<TypeParam, 3>(af::dim4(32, 32, 32));
}

TYPED_TEST(FFT_REAL, Odd3D)
{
    fft_real<TypeParam, 3>(af::dim4(25, 32, 32));
}
