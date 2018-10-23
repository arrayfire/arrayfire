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

TEST(Approx1, CPPUsage)
{
    //! [ex_signal_approx1]

    // Input data array.
    float input_vals[3] = {10.0, 20.0, 30.0};
    array in(dim4(3, 1), input_vals);
    // [3 1 1 1]
    //     10.0000
    //     20.0000
    //     30.0000

    // Array of positions to be found along the first dimension.
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Perform interpolation across dimension 0.
    array interp = approx1(in, pos);
    // [5 1 1 1]
    //     10.0000
    //     15.0000
    //     20.0000
    //     25.0000
    //     30.0000

    //! [ex_signal_approx1]

    float civ[5] = {10.0, 15.0, 20.0, 25.0, 30.0};
    array interp_gold(dim4(5,1), civ);
    ASSERT_ARRAYS_EQ(interp, interp_gold);

}


TEST(Approx1, CPPUniformUsage)
{
    //! [ex_signal_approx1_uniform]

    float input_vals[9] = {10.0, 20.0, 30.0,
                           40.0, 50.0, 60.0,
                           70.0, 80.0, 90.0};
    array in(dim4(3, 3), input_vals);
    // [3 3 1 1]
    //     10.0000    40.0000    70.0000
    //     20.0000    50.0000    80.0000
    //     30.0000    60.0000    90.0000

    // Array of positions to be found along the interpolation
    // dimension, `interp_dim`.
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Define range of indices with which the input values will
    // correspond along the interpolation dimension.
    const double idx_start = 0.0;
    const double idx_step = 1.0;

    // Perform interpolation across dimension 0.
    int interp_dim = 0;
    array col_major_interp = approx1(in, pos, interp_dim, idx_start, idx_step);
    // [5 3 1 1]
    //     10.0000    40.0000    70.0000
    //     15.0000    45.0000    75.0000
    //     20.0000    50.0000    80.0000
    //     25.0000    55.0000    85.0000
    //     30.0000    60.0000    90.0000

    // Perform interpolation across dimension 1.
    interp_dim = 1;
    array row_major_interp = approx1(in, transpose(pos), interp_dim, idx_start, idx_step);
    // [3 5 1 1]
    //     10.0000    25.0000    40.0000    55.0000    70.0000
    //     20.0000    35.0000    50.0000    65.0000    80.0000
    //     30.0000    45.0000    60.0000    75.0000    90.0000

    //! [ex_signal_approx1_uniform]

    float civ[15] = {10.0, 15.0, 20.0, 25.0, 30.0,
                     40.0, 45.0, 50.0, 55.0, 60.0,
                     70.0, 75.0, 80.0, 85.0, 90.0};
    array interp_gold_col(dim4(5,3), civ);
    ASSERT_ARRAYS_EQ(col_major_interp, interp_gold_col);


    float riv[15] = {10.0, 20.0, 30.0,
                     25.0, 35.0, 45.0,
                     40.0, 50.0, 60.0,
                     55.0, 65.0, 75.0,
                     70.0, 80.0, 90.0};
    array interp_gold_row(dim4(3,5), riv);
    ASSERT_ARRAYS_EQ(row_major_interp, interp_gold_row);
}

TEST(Approx1, CPPDecimalStepRescaleGrid)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[5] = {0, 0.25, 0.5, 0.75, 1.0};
    array pos(dim4(5,1), pv);

    const int interp_grid_start = 0;
    const double interp_grid_step = 0.5;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {10.0, 15.0, 20.0, 25.0, 30.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPRepeatPos)
{
    float inv[9] = {10.0, 20.0, 30.0,
                    40.0, 50.0, 60.0,
                    70.0, 80.0, 90.0};
    array in(dim4(3, 3), inv);
    float pv[5] = {0.0, 0.5, 0.5, 1.5, 1.5};
    array pos(dim4(5,1), pv);

    const int interp_grid_start = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[15] = {10.0, 15.0, 15.0, 25.0, 25.0,
                    40.0, 45.0, 45.0, 55.0, 55.0,
                    70.0, 75.0, 75.0, 85.0, 85.0};
    array interp_gold(dim4(5,3), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}


TEST(Approx1, CPPNonMonotonicPos)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[5] = {0.5, 1.0, 1.5, 0.0, 2.0};
    array pos(dim4(5,1), pv);

    const int interp_grid_start = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {15.0, 20.0, 25.0, 10.0, 30.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPMismatchingIndexingDim)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[4] = {0.0, 0.5, 1.0, 2.0};
    array pos(dim4(1,4), pv);

    const int interp_grid_start = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim = 1;
    const float off_grid = -1.0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step,
                           AF_INTERP_LINEAR, off_grid);

    float iv[12] = {10.0, 20.0, 30.0,
                   -1.0, -1.0, -1.0,
                   -1.0, -1.0, -1.0,
                   -1.0, -1.0, -1.0};
    array interp_gold(dim4(3,4), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPNegativeGridStart)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(5,1), pv);

    const int interp_grid_start = -1;
    const double interp_grid_step = 1;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {20.0, 25.0, 30.0, 0.0, 0.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);

}

TEST(Approx1, CPPInterpolateBackwards)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    array pos(dim4(3,1), pv);

    const int interp_grid_start = in.elements()-1;
    const double interp_grid_step = -1;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {30.0, 25.0, 20.0, 15.0, 10.0};
    array interp_gold(dim4(3,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}


TEST(Approx1, CPPStartOffGridAndNegativeStep)
{
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3,1), inv);
    float pv[5] = {0.0, -0.5, -1.0, -1.5, -2.0};
    array pos(dim4(5,1), pv);

    const int interp_grid_start = -1;
    const double interp_grid_step = -1;
    const int interp_dim = 0;
    array interp = approx1(in,
                           pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {0.0, 0.0, 10.0, 15.0, 20.0};
    array interp_gold(dim4(5,1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPUniformInvalidStepSize)
{
    try
    {
        float inv[3] = {10.0, 20.0, 30.0};
        array in(dim4(3,1), inv);
        float pv[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
        array pos(dim4(5,1), pv);

        const int interp_grid_start = 0;
        const double interp_grid_step = 0;
        const int interp_dim = 0;
        array interp = approx1(in,
                               pos, interp_dim, interp_grid_start, interp_grid_step);
        FAIL() << "Expected af::exception\n";
    } catch (af::exception &ex) {
        SUCCEED();
    } catch(...) {
        FAIL() << "Expected af::exception\n";
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
    array interp = approx1(in, pos);
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyInput)
{
    array in;
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3,1), pv);

    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyPosAndInput)
{
    array in;
    array pos;
    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, UseNullInitialOutput) {
    float h_in[3] = {10, 20, 30};
    dim_t h_in_dims[1] = {3};

    af_array in = 0;
    af_create_array(&in, &h_in[0], 1, &h_in_dims[0], f32);

    float h_pos[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    dim_t h_pos_dims[1] = {5};
    af_array pos = 0;
    af_create_array(&pos, &h_pos[0], 1, &h_pos_dims[0], f32);

    dim_t h_out_dims[1] = {5};
    af_array out = 0;
    af_approx1(&out, in, pos, AF_INTERP_LINEAR, 0);

    ASSERT_FALSE(out == 0);
}

TEST(Approx1, UseExistingOutputArray) {
    float h_in[3] = {10, 20, 30};
    dim_t h_in_dims[1] = {3};

    af_array in = 0;
    af_create_array(&in, &h_in[0], 1, &h_in_dims[0], f32);

    float h_pos[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    dim_t h_pos_dims[1] = {5};
    af_array pos = 0;
    af_create_array(&pos, &h_pos[0], 1, &h_pos_dims[0], f32);

    dim_t h_out_dims[1] = {5};
    af_array out_ptr = 0;
    af_create_handle(&out_ptr, 1, &h_out_dims[0], f32);
    // af_print_array_gen("out_ptr", out_ptr, 4);
    af_array out_ptr_copy = out_ptr;
    // af_print_array_gen("out_ptr_copy", out_ptr_copy, 4);
    af_approx1(&out_ptr, in, pos, AF_INTERP_LINEAR, 0);
    // af_print_array_gen("out_ptr", out_ptr, 4);
    // af_print_array_gen("out_ptr_copy", out_ptr_copy, 4);

    ASSERT_EQ(out_ptr_copy, out_ptr);
    ASSERT_ARRAYS_EQ(out_ptr_copy, out_ptr);
}

TEST(Approx1, UseExistingOutputSlice) {
    float h_in[3] = {10, 20, 30};
    dim_t h_in_dims[1] = {3};

    af_array in = 0;
    af_create_array(&in, &h_in[0], 1, &h_in_dims[0], f32);

    float h_pos[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    dim_t h_pos_dims[1] = {5};
    af_array pos = 0;
    af_create_array(&pos, &h_pos[0], 1, &h_pos_dims[0], f32);

    float h_out[15] = {1.0, 1.5, 2.0, 2.5, 3.0,
                       4.0, 4.5, 5.0, 5.5, 6.0,
                       7.0, 7.5, 8.0, 8.5, 9.0};
    dim_t h_out_dims[2] = {5, 3};
    af_array out = 0;
    af_create_array(&out, &h_out[0], 2, &h_out_dims[0], f32);
    // af_print_array_gen("out", out, 4);
    af_seq idx_dim1 = {1, 1, 1}; // get slice 1 of dim1
    af_seq idx[2] = {af_span, idx_dim1};
    af_array out_slice = 0;
    af_index(&out_slice, out, 2, &idx[0]);
    // af_print_array_gen("out_slice", out_slice, 4);
    af_approx1(&out_slice, in, pos, AF_INTERP_LINEAR, 0);
    // af_print_array_gen("out_slice", out_slice, 4);
    // af_print_array_gen("out", out, 4);

    dim_t nelems = 0;
    af_get_elements(&nelems, out);
    vector<float> h_out_approx(nelems);
    af_get_data_ptr(&h_out_approx.front(), out);

    float h_gold_arr[5] = {10.0, 15.0, 20.0, 25.0, 30.0};

    // Check slice 1 of dim1 (elements 5 to 9 of h_out_approx) to see if they
    // contain af_approx1's results
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(h_gold_arr[i], h_out_approx[5+i]) << "at i: " << i << endl;
    }
}
