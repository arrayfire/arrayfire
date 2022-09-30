/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::exception;
using af::identity;
using af::matmul;
using af::max;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

///////////////////////////////// CPP ////////////////////////////////////
TEST(QRFactorized, CPP) {
    if (noLAPACKTests()) return;

    int resultIdx = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/lapack/qrfactorized.test"),
                                   numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));

    array q, r, tau;
    qr(q, r, tau, input);

    dim4 qdims = q.dims();
    dim4 rdims = r.dims();

    // Get result
    float* qData = new float[qdims.elements()];
    q.host((void*)qData);
    float* rData = new float[rdims.elements()];
    r.host((void*)rData);

    // Compare result
    for (int y = 0; y < (int)qdims[1]; ++y) {
        for (int x = 0; x < (int)qdims[0]; ++x) {
            int elIter = y * qdims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], qData[elIter], 0.001)
                << "at: " << elIter << endl;
        }
    }

    resultIdx = 1;

    for (int y = 0; y < (int)rdims[1]; ++y) {
        for (int x = 0; x < (int)rdims[0]; ++x) {
            // Test only upper half
            if (x <= y) {
                int elIter = y * rdims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], rData[elIter], 0.001)
                    << "at: " << elIter << endl;
            }
        }
    }

    // Delete
    delete[] qData;
    delete[] rData;
}

template<typename T>
void qrTester(const int m, const int n, double eps) {
    try {
        SUPPORTED_TYPE_CHECK(T);
        if (noLAPACKTests()) return;

#if 1
        array in = cpu_randu<T>(dim4(m, n));
#else
        array in = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

        //! [ex_qr_unpacked]
        array q, r, tau;
        qr(q, r, tau, in);
        //! [ex_qr_unpacked]

        array qq = matmul(q, q.H());
        array ii = identity(qq.dims(), qq.type());

        ASSERT_NEAR(0, max<double>(abs(real(qq - ii))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(qq - ii))), eps);

        //! [ex_qr_recon]
        array re = matmul(q, r);
        //! [ex_qr_recon]

        ASSERT_NEAR(0, max<double>(abs(real(re - in))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(re - in))), eps);

        //! [ex_qr_packed]
        array out = in.copy();
        array tau2;
        qrInPlace(tau2, out);
        //! [ex_qr_packed]

        array r2 = upper(out);

        ASSERT_NEAR(0, max<double>(abs(real(tau - tau2))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(tau - tau2))), eps);

        ASSERT_NEAR(0, max<double>(abs(real(r2 - r))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(r2 - r))), eps);

    } catch (exception& ex) {
        cout << ex.what() << endl;
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
class QR : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(QR, TestTypes);

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

TEST(QR, InPlaceNullOutput) {
    if (noLAPACKTests()) return;
    dim4 dims(3, 3);
    af_array in = 0;
    ASSERT_SUCCESS(af_randu(&in, dims.ndims(), dims.get(), f32));

    ASSERT_EQ(AF_ERR_ARG, af_qr_inplace(NULL, in));
    ASSERT_SUCCESS(af_release_array(in));
}
