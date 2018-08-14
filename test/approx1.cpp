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
    const af_interp_type method = AF_INTERP_LINEAR;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx1.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    array input(idims, &(in[0].front()));
    array pos(pdims, &(in[1].front()));

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

TEST(Approx1, OtherDimLinear)
{
    int start = 0;
    int stop = 10000;
    int step = 100;
    int num = 1000;
    af::array xi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array yi = 4 * xi - 3;
    af::array xo = af::round(step * af::randu(num, 2, 2, 2));
    af::array yo = 4 * xo - 3;
    for (int d = 1; d < 4; d++) {
        af::dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        af::array yi_reordered = af::reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array xo_reordered = af::reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array yo_reordered = af::approx1(yi_reordered, xo_reordered,
                                             d, start, step, AF_INTERP_LINEAR);
        rdims[d] = 0;
        rdims[0] = d;
        af::array res = af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, OtherDimCubic)
{
    float start = 0;
    float stop = 100;
    float step = 0.01;
    int num = 1000;
    af::array xi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array yi = af::sin(xi);
    af::array xo = af::round(step * af::randu(num, 2, 2, 2));
    af::array yo = af::sin(xo);
    for (int d = 1; d < 4; d++) {
        af::dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        af::array yi_reordered = af::reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array xo_reordered = af::reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array yo_reordered = af::approx1(yi_reordered, xo_reordered,
                                             d, start, step, AF_INTERP_CUBIC);
        rdims[d] = 0;
        rdims[0] = d;
        af::array res = af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}
