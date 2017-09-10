/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::cfloat;
using af::cdouble;

///////////////////////////////// CPP ////////////////////////////////////
TEST(QRFactorized, CPP)
{
    if (noDoubleTests<float>()) return;
    if (noLAPACKTests()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/qrfactorized.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));

    af::array q, r, tau;
    af::qr(q, r, tau, input);

    af::dim4 qdims = q.dims();
    af::dim4 rdims = r.dims();

    // Get result
    float* qData = new float[qdims.elements()];
    q.host((void*)qData);
    float* rData = new float[rdims.elements()];
    r.host((void*)rData);

    // Compare result
    for (int y = 0; y < (int)qdims[1]; ++y) {
        for (int x = 0; x < (int)qdims[0]; ++x) {
            int elIter = y * qdims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], qData[elIter], 0.001) << "at: " << elIter << std::endl;
        }
    }

    resultIdx = 1;

    for (int y = 0; y < (int)rdims[1]; ++y) {
        for (int x = 0; x < (int)rdims[0]; ++x) {
            // Test only upper half
            if(x <= y) {
                int elIter = y * rdims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], rData[elIter], 0.001) << "at: " << elIter << std::endl;
            }
        }
    }

    // Delete
    delete[] qData;
    delete[] rData;
}

template<typename T>
void qrTester(const int m, const int n, double eps)
{
    try {
        if (noDoubleTests<T>()) return;
        if (noLAPACKTests()) return;

#if 1
        af::array in = cpu_randu<T>(af::dim4(m, n));
#else
        af::array in = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

        //! [ex_qr_unpacked]
        af::array q, r, tau;
        af::qr(q, r, tau, in);
        //! [ex_qr_unpacked]

        af::array qq = af::matmul(q, q.H());
        af::array ii = af::identity(qq.dims(), qq.type());

        ASSERT_NEAR(0, af::max<double>(af::abs(real(qq - ii))), eps);
        ASSERT_NEAR(0, af::max<double>(af::abs(imag(qq - ii))), eps);

        //! [ex_qr_recon]
        af::array re = af::matmul(q, r);
        //! [ex_qr_recon]

        ASSERT_NEAR(0, af::max<double>(af::abs(real(re - in))), eps);
        ASSERT_NEAR(0, af::max<double>(af::abs(imag(re - in))), eps);

        //! [ex_qr_packed]
        af::array out = in.copy();
        af::array tau2;
        qrInPlace(tau2, out);
        //! [ex_qr_packed]

        af::array r2 = upper(out);

        ASSERT_NEAR(0, af::max<double>(af::abs(real(tau - tau2))), eps);
        ASSERT_NEAR(0, af::max<double>(af::abs(imag(tau - tau2))), eps);

        ASSERT_NEAR(0, af::max<double>(af::abs(real(r2 - r))), eps);
        ASSERT_NEAR(0, af::max<double>(af::abs(imag(r2 - r))), eps);


    } catch(af::exception &ex) {
        std::cout << ex.what() << std::endl;
        throw;
    }
}

template<typename T>
double eps();

template<>
double eps<float>() {
    return 1e-3;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 1e-3;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}
template<typename T>
class QR : public ::testing::Test
{

};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(QR, TestTypes);

TYPED_TEST(QR, RectangularLarge0) {
    qrTester<TypeParam>(1000, 500, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularMultipleOfTwoLarge0) {
    qrTester<TypeParam>(1024, 512, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularLarge1) {
    qrTester<TypeParam>(500, 1000, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularMultipleOfTwoLarge1) {
    qrTester<TypeParam>(512, 1024, eps<TypeParam>());
}
