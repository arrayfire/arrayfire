/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/complex.h>
#include <af/dim4.hpp>
#include <af/index.h>
#include <af/signal.h>
#include <af/traits.hpp>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <complex>
#include <string>
#include <vector>

using af::abs;
using af::approx1;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::reorder;
using af::span;
using af::seq;
using af::sum;

using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Approx1 : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// Create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// Register the type list
TYPED_TEST_CASE(Approx1, TestTypes);

template<typename T>
void approx1Test(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray = 0;
    af_array posArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

        ASSERT_SUCCESS(af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<BT>::af_type));

    ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Nearest)
{
    approx1Test<TypeParam>(string(TEST_DIR"/approx/approx1.test"), 0, AF_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1Linear)
{
    approx1Test<TypeParam>(string(TEST_DIR"/approx/approx1.test"), 1, AF_INTERP_LINEAR);
}


template<typename T>
void approx1CubicTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray = 0;
    af_array posArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;

    float max = real(outData[0]), min = real(outData[0]);
    for(int i=1; i < (int)nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        double integral;
        // Test that control points are exact
        if((std::modf(in[1][elIter], &integral) < 0.001) || (std::modf(in[1][elIter], &integral) > 0.999))  {
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
        } else {
            // Match intermediate values within a threshold
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.035 * range;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Cubic)
{
    approx1CubicTest<TypeParam>(string(TEST_DIR"/approx/approx1_cubic.test"), 0, AF_INTERP_CUBIC_SPLINE);
}

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx1ArgsTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, const af_err err)
{
    if (noDoubleTests<T>()) return;
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<BT>::af_type));

    ASSERT_EQ(err, af_approx1(&outArray, inArray, posArray, method, 0));

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPos2D)
{
    approx1ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx1_pos2d.test"), 0, AF_INTERP_NEAREST, AF_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1LinearArgsPos2D)
{
    approx1ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx1_pos2d.test"), 1, AF_INTERP_LINEAR, AF_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1ArgsInterpBilinear)
{
    approx1ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx1.test"), 0, AF_INTERP_BILINEAR, AF_ERR_ARG);
}

template<typename T>
void approx1ArgsTestPrecision(string pTestFile, const unsigned resultIdx, const af_interp_type method)
{
    if (noDoubleTests<T>()) return;
    vector<dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    af_array inArray  = 0;
    af_array posArray = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&posArray, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<T>::af_type));

    if((af_dtype) dtype_traits<T>::af_type == c32 ||
       (af_dtype) dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx1(&outArray, inArray, posArray, method, 0));
    } else {
        ASSERT_SUCCESS(af_approx1(&outArray, inArray, posArray, method, 0));
    }

    if(inArray   != 0) af_release_array(inArray);
    if(posArray  != 0) af_release_array(posArray);
    if(outArray  != 0) af_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPrecision)
{
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/approx1.test"), 0, AF_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1LinearArgsPrecision)
{
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/approx1.test"), 1, AF_INTERP_LINEAR);
}

TYPED_TEST(Approx1, Approx1CubicArgsPrecision)
{
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/approx1_cubic.test"), 2, AF_INTERP_CUBIC_SPLINE);
}


//////////////////////////////////////// CPP //////////////////////////////////
//
TEST(Approx1, CPP)
{
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx1.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    array input(idims, &(in[0].front()));
    array pos(pdims, &(in[1].front()));
    const af_interp_type method = AF_INTERP_LINEAR;
    array output = approx1(input, pos, method, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx1, CPPNearestBatch)
{
    array input = randu(600, 10);
    array pos   = input.dims(0) * randu(100, 10);

    array outBatch = approx1(input, pos, AF_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) = approx1(input(span, i),
                                     pos(span, i),
                                     AF_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) = approx1(input(span, i),
                                   pos(span, i),
                                   AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPLinearBatch)
{
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) = approx1(input(span, i),
                                     pos(span, i),
                                     AF_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) = approx1(input(span, i),
                                   pos(span, i),
                                   AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPCubicBatch)
{
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, AF_INTERP_CUBIC_SPLINE);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) = approx1(input(span, i),
                                     pos(span, i),
                                     AF_INTERP_CUBIC_SPLINE);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) = approx1(input(span, i),
                                   pos(span, i),
                                   AF_INTERP_CUBIC_SPLINE);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPNearestMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;
    array input = randu(1, largeDim);
    array pos   = input.dims(0) * randu(1, largeDim);
    array out   = approx1(input, pos, AF_INTERP_NEAREST);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    out   = approx1(input, pos, AF_INTERP_NEAREST);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    out   = approx1(input, pos, AF_INTERP_NEAREST);

    SUCCEED();
}

TEST(Approx1, CPPLinearMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;
    array input = iota(dim4(1, largeDim), c32);
    array pos   = input.dims(0) * randu(1, largeDim);
    array outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    input = iota(dim4(1, 1, largeDim), c32);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    input = iota(dim4(1, 1, 1, largeDim), c32);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_LINEAR);

    SUCCEED();
}

TEST(Approx1, CPPCubicMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;
    array input = iota(dim4(1, largeDim), c32);
    array pos   = input.dims(0) * randu(1, largeDim);
    array outBatch = approx1(input, pos, AF_INTERP_CUBIC);

    input = iota(dim4(1, 1, largeDim), c32);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_CUBIC);

    input = iota(dim4(1, 1, 1, largeDim), c32);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, AF_INTERP_CUBIC);

    SUCCEED();
}

TEST(Approx1, CPPUsage)
{
    //! [ex_signal_approx1]

    // Input data.
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3, 1), inv);

    // Positions of interpolated values.
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);

    // Interpolate data across the first dimension, with a start grid
    // index of 0 and a grid step size of 1.
    array interp_uniform = approx1(in,
                                   pos, 0,
                                   0, 1);
    // interp_uniform == { 10, 15, 20, 25, 30 };

    // Interpolate data across the first dimension. Grid start point
    // is 0 and grid step is 1 by default.
    array interp         = approx1(in, pos);
    // interp == interp_uniform == { 10, 15, 20, 25, 30 };

    //! [ex_signal_approx1]

    float iv[5] = {10.0, 15.0, 20.0, 25.0, 30.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
    ASSERT_ARRAYS_EQ(interp, interp_uniform);
}

TEST(Approx1, CPP2DInput)
{
    float inv[9] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0,
                    70.0, 80.0, 90.0};
    array in(dim4(3,3), inv);
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);

    array interpolated = approx1(in, pos);
    float iv[15] = {10.0, 15.0, 20.0, 25.0, 30.0,
                    40.0, 45.0, 50.0, 55.0, 60.0,
                    70.0, 75.0, 80.0, 85.0, 90.0};
    array interp_gold(dim4(5,3), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformUsage)
{
    //! [ex_uniform_approx1]

    // Input data.
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);

    // Positions of interpolated values.
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);

    const int    dim0 = 0;  // Dimension to interpolate across.
    const int   start = 0;  // Starting index value of grid on which measurements were made.
    const double step = 1;  // Step size of grid on which measurements were made.
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    // interpolated == {10, 15, 20, 25, 30};
    //! [ex_uniform_approx1]

    float iv[5] = {10.0, 15.0, 20.0, 25.0, 30.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformNonMonotonicPos)
{
    float inv[10] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    array in(dim4(10,1), inv);

    float pv[10] = {1.0, 0.0, 2.0, 9.0, 7.0, 4.0, 3.0, 5.0, 6.0, 8.0};
    array pos(dim4(10,1), pv);

    const int start = 0;
    const double step = 1.0;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[10] = {20.0, 10.0, 30.0, 100.0, 80.0, 50.0, 40.0, 60.0, 70.0, 90.0};
    array interp_gold(dim4(10,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformSamePos)
{
    float inv[10] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    array in(dim4(10,1), inv);

    float pv[10] = {0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0, 8.0};
    array pos(dim4(10,1), pv);

    const int start = 0;
    const double step = 1.0;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[10] = {10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 30.0, 30.0, 30.0, 90.0};
    array interp_gold(dim4(10,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformDecimalStep)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0, 0.5, 1.0};
    array pos(dim4(3,1), pv);

    const int start = 0;
    const double step = 0.5;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[3] = {10.0, 20.0, 30.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformMismatchingIndexingDim)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    const int start = 0;
    const double step = 1.0;
    const int dim1 = 1;
    array interpolated = approx1(in,
                                 pos, dim1,
                                 start, step);
    float iv[3] = {10.0, 0.0, 0.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformPositiveStartAndStep)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    const int start = 1;
    const double step = 1;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[3] = {0.0, 10.0, 20.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformNegativeStart)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    const int start = -1;
    const double step = 1;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[3] = {20.0, 30.0, 0.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformInterpolateBackwards)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    const int start = 2;
    const double step = -1;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[3] = {30.0, 20.0, 10.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformOffGridAndNegativeStep)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[3] = {0.0, -1.0, -2.0};
    array pos(dim4(3,1), pv);

    const int start = -1;
    const double step = -1;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[3] = {0.0, 10.0, 20.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformInvalidStepSize)
{
    try
    {
        float inv[3] = {10.0, 20.0, 30.0};
        array in(dim4(3,1), inv);
        float pv[3] = {0.0, -1.0, -2.0};
        array pos(dim4(3,1), pv);

        const int start = -1;
        const double step = 0;
        const int dim0 = 0;
        array interpolated = approx1(in,
                                     pos, dim0,
                                     start, step);
        FAIL() << "Expected af::exception\n";
        float iv[3] = {0.0, 10.0, 20.0};
        array interp_gold(dim4(3,1), iv);
        ASSERT_ARRAYS_EQ(interpolated, interp_gold);
    } catch (af::exception &ex) {
        SUCCEED();
    } catch(...) {
        FAIL() << "Expected af::exception\n";
    }
}

TEST(Approx1, CPPUniformInterpDim0)
{
    float inv[9] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0,
                    70.0, 80.0, 90.0};
    array in(dim4(3,3), inv);
    float pv[3] = {0, 1, 2};
    array pos(dim4(3,1), pv);

    const int start = 0;
    const double step = 1;
    const int dim0 = 0;
    array interpolated = approx1(in,
                                 pos, dim0,
                                 start, step);
    float iv[9] = {10.0, 20.0, 30.0,
                   40.0, 50.0, 60.0,
                   70.0, 80.0, 90.0};
    array interp_gold(dim4(3,3), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, CPPUniformInterpDim1)
{
    float inv[9] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0,
                    70.0, 80.0, 90.0};
    array in(dim4(3,3), inv);
    float pv[3] = {0, 1, 2};
    array pos(dim4(1,3), pv);

    const int start = 0;
    const double step = 1;
    const int dim1 = 1;
    array interpolated = approx1(in,
                                 pos, dim1,
                                 start, step);
    float iv[9] = {10.0, 20.0, 30.0,
                   40.0, 50.0, 60.0,
                   70.0, 80.0, 90.0};
    array interp_gold(dim4(3,3), iv);
    ASSERT_ARRAYS_EQ(interpolated, interp_gold);
}

TEST(Approx1, OtherDimLinear)
{
    int start = 0;
    int stop = 10000;
    int step = 100;
    int num = 1000;
    array xi = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi = 4 * xi - 3;
    array xo = af::round(step * randu(num, 2, 2, 2));
    array yo = 4 * xo - 3;
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered = reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered = reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered,
                                     d, start, step, AF_INTERP_LINEAR);
        array res = reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, OtherDimCubic)
{
    float start = 0;
    float stop = 100;
    float step = 0.01;
    int num = 1000;
    array xi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    array yi = af::sin(xi);
    array xo = af::round(step * af::randu(num, 2, 2, 2));
    array yo = af::sin(xo);
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered = reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered = reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered,
                                     d, start, step, AF_INTERP_CUBIC);
        array res = reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

// Unless the sampling grid specifications - begin, step - are
// specified by the user, ArrayFire will assume a regular grid with a
// starting index of 0 and a step value of 1.
TEST(Approx1, CPPInfCheck)
{
    array sampled(seq(0.0, 5.0, 0.5));
    sampled(0) = af::Inf;
    seq xo(0.0, 2.0, 0.25);
    array interp = approx1(sampled, xo);
    array interp_augmented = join(1, xo, interp);

    float goldv[9] = {af::Inf, af::Inf, af::Inf, af::Inf, 0.5, 0.625, 0.75, 0.875, 1.0};
    array gold(dim4(9,1), goldv);
    interp(af::isInf(interp)) = 0;
    gold(af::isInf(gold)) = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPUniformInfCheck)
{
    array sampled(seq(10.0, 50.0, 10.0));
    sampled(0) = af::Inf;
    seq xo(0.0, 8.0, 2.0);
    array interp = approx1(sampled,
                           xo, 0,
                           0, 2);
    float goldv[5] = {af::Inf, 20.0, 30.0, 40.0, 50.0};
    array gold(dim4(5,1), goldv);
    interp(af::isInf(interp)) = 0;
    gold(af::isInf(gold)) = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPEmptyPos)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    array pos;
    array interpolated = approx1(in, pos);
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

TEST(Approx1, CPPEmptyInput)
{
    array in;
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    array interpolated = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

TEST(Approx1, CPPEmptyPosAndInput)
{
    array in;
    array pos;
    array interpolated = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

