/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/signal.h>
#include <af/traits.hpp>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <string>
#include <vector>

using af::abs;
using af::approx2;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::seq;
using af::span;
using af::sum;

using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Approx2 : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// register the type list
TYPED_TEST_CASE(Approx2, TestTypes);

template<typename T>
void approx2Test(string pTestFile, const unsigned resultIdx, const af_interp_type method, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

        ASSERT_SUCCESS(af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) dtype_traits<BT>::af_type));

    ASSERT_SUCCESS(af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx2, Approx2Nearest)
{
    approx2Test<TypeParam>(string(TEST_DIR"/approx/approx2.test"), 0, AF_INTERP_NEAREST);
}

TYPED_TEST(Approx2, Approx2Linear)
{
    approx2Test<TypeParam>(string(TEST_DIR"/approx/approx2.test"), 1, AF_INTERP_LINEAR);
}
TYPED_TEST(Approx2, NearestBatch)
{
    approx2Test<TypeParam>(string(TEST_DIR"/approx/approx2_batch.test"), 0, AF_INTERP_NEAREST);
}
TYPED_TEST(Approx2, LinearBatch)
{
    approx2Test<TypeParam>(string(TEST_DIR"/approx/approx2_batch.test"), 1, AF_INTERP_LINEAR);
}

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx2ArgsTest(string pTestFile, const unsigned resultIdx, const af_interp_type method, const af_err err)
{
    if (noDoubleTests<T>()) return;
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) dtype_traits<BT>::af_type));

    ASSERT_EQ(err, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
}

TYPED_TEST(Approx2, Approx2NearestArgsPos3D)
{
    approx2ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx2_pos3d.test"), 0, AF_INTERP_NEAREST, AF_ERR_SIZE);
}

TYPED_TEST(Approx2, Approx2LinearArgsPos3D)
{
    approx2ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx2_pos3d.test"), 1, AF_INTERP_LINEAR, AF_ERR_SIZE);
}

TYPED_TEST(Approx2, Approx2NearestArgsPosUnequal)
{
    approx2ArgsTest<TypeParam>(string(TEST_DIR"/approx/approx2_unequal.test"), 0, AF_INTERP_NEAREST, AF_ERR_SIZE);
}

template<typename T>
void approx2ArgsTestPrecision(string pTestFile, const unsigned resultIdx, const af_interp_type method)
{
    if (noDoubleTests<T>()) return;
    vector<dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) dtype_traits<T>::af_type));


    if((af_dtype) dtype_traits<T>::af_type == c32 ||
       (af_dtype) dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    } else {
        ASSERT_SUCCESS(af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    }

    if(inArray   != 0) af_release_array(inArray);
    if(pos0Array != 0) af_release_array(pos0Array);
    if(pos1Array != 0) af_release_array(pos1Array);
    if(outArray  != 0) af_release_array(outArray);
}

#define APPROX2_ARGSP(desc, file, resultIdx, method)                                    \
    TYPED_TEST(Approx2, desc)                                                           \
    {                                                                                   \
        approx2ArgsTestPrecision<TypeParam>(string(TEST_DIR"/approx/"#file".test"),     \
                                            resultIdx, method);                         \
    }

    APPROX2_ARGSP(Approx2NearestArgsPrecision, approx2, 0, AF_INTERP_NEAREST);
    APPROX2_ARGSP(Approx2LinearArgsPrecision, approx2, 1, AF_INTERP_LINEAR);


//////////////////////////////////// CPP ////////////////////////////////////
//
TEST(Approx2, CPP)
{
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx2.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims,&(in[0].front()));
    array pos0(pdims,&(in[1].front()));
    array pos1(qdims,&(in[2].front()));
    array output = approx2(input, pos0, pos1, AF_INTERP_LINEAR, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2Cubic, CPP)
{
    const unsigned resultIdx = 0;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<float> > tests;
    readTests<BT, float, float>(string(TEST_DIR"/approx/approx2_cubic.test"),numDims,in,tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims,&(in[0].front()));
    input = input.T();
    array pos0(pdims,&(in[1].front()));
    array pos1(qdims,&(in[2].front()));
    pos0 = tile(pos0, 1, pos0.dims(0));
    pos1 = tile(pos1.T(), pos1.dims(0));
    array output = approx2(input, pos0, pos1, AF_INTERP_BICUBIC_SPLINE, 0).T();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

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
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.01 * range);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2, CPPNearestBatch)
{
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx2, CPPLinearBatch)
{
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) = approx2(input(span, span, i),
            pos(span, span, i), qos(span, span, i), AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx2, CPPNearestMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;

    array input = randu(1, largeDim);
    array pos   = input.dims(0) * randu(1, 10);
    array qos   = input.dims(1) * randu(1, 10);
    array out   = approx2(input, pos, qos, AF_INTERP_NEAREST);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_NEAREST);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_NEAREST);

    SUCCEED();
}

TEST(Approx2, CPPLinearMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;

    array input = randu(1, largeDim);
    array pos   = input.dims(0) * randu(1, 10);
    array qos   = input.dims(1) * randu(1, 10);
    array out   = approx2(input, pos, qos, AF_INTERP_LINEAR);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_LINEAR);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_LINEAR);

    SUCCEED();
}

TEST(Approx2, CPPCubicMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;

    array input = randu(1, largeDim);
    array pos   = input.dims(0) * randu(1, 10);
    array qos   = input.dims(1) * randu(1, 10);
    array out   = approx2(input, pos, qos, AF_INTERP_BICUBIC);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_BICUBIC);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * randu(1, 1, 1, largeDim);
    out   = approx2(input, pos, qos, AF_INTERP_BICUBIC);

    SUCCEED();
}

TEST(Approx2, CPPUsage)
{
    //! [ex_signal_approx2]

    // constant input data
    // {{1 2 3},
    //  {1 2 3},
    //  {1 2 3}},
    float input_vals[9] = {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3};
    array input(3, 3, input_vals);

    // generate grid of interpolation locations
    // interpolation locations along dim0
    float p0[4] = {0.5, 1.5,
                   0.5, 1.5};
    array pos0(2, 2, p0);
    // interpolation locations along dim1
    float p1[4] = {0.5, 0.5,
                   1.5, 1.5};
    array pos1(2, 2, p1);

    array interpolated = approx2(input, pos0, pos1);
    // interpolated == {{1.5 2.5},
    //                  {1.5 2.5}};

    //! [ex_signal_approx2]

    float expected_interp[4] = {1.5, 1.5,
                                2.5, 2.5};

    array interpolated_gold(2, 2, expected_interp);
    ASSERT_ARRAYS_NEAR(interpolated, interpolated_gold, 1e-5);
}

TEST(Approx2, CPPUniformUsage)
{
    //! [ex_uniform_approx2]

    // constant input data
    // {{1 2 3},
    //  {1 2 3},
    //  {1 2 3}},
    float input_vals[9] = {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3};
    array input(3, 3, input_vals);

    // generate grid of interpolation locations
    // interpolation locations along dim0
    float p0[4] = {0.5, 1.5,
                   0.5, 1.5};
    array pos0(2, 2, p0);
    // interpolation locations along dim1
    float p1[4] = {0.5, 0.5,
                   1.5, 1.5};
    array pos1(2, 2, p1);

    const int start = 0;
    const double step = 1;
    const int dim0 = 0;
    array interpolated = approx2(input,
                                 pos0, dim0,
                                 pos1, dim0 + 1,
                                 start, step,
                                 start, step);
    // interpolated == {{1.5 2.5},
    //                  {1.5 2.5}};
    //! [ex_uniform_approx2]

    float expected_interp[4] = {1.5, 1.5,
                                2.5, 2.5};

    array interpolated_gold(2, 2, expected_interp);
    ASSERT_ARRAYS_NEAR(interpolated, interpolated_gold, 1e-5);
}

TEST(Approx2, CPPUniformOneDimIndices)
{
    float inv[9] = {10, 20, 30,
                    40, 50, 60,
                    70, 80, 90};
    array input(dim4(3,3), inv);

    // generate grid of interpolation locations
    float p0[3] = {0, 1, 2};
    float p1[3] = {0, 1, 2};
    array pos0(dim4(3,1), p0);
    array pos1(dim4(3,1), p1);

    const int start = 0;
    const double step = 1;
    const int d0 = 0;
    array interpolated = approx2(input,
                                 pos0, d0,
                                 pos1, d0 + 1,
                                 start, step,
                                 start, step);

    const float expected_interp[3] = {10, 50, 90};


    array interpolated_gold(dim4(3,1), expected_interp);
    ASSERT_ARRAYS_NEAR(interpolated, interpolated_gold, 1e-5);
}

TEST(Approx2, CPPUniformTwoDimIndices)
{
    float inv[9] = {10, 20, 30,
                    40, 50, 60,
                    70, 80, 90};
    array input(dim4(3,3), inv);

    // generate grid of interpolation locations
    float p0[4] = {0, 2, 0, 2};
    float p1[4] = {0, 0, 2, 2};
    array pos0(dim4(2,2), p0);
    array pos1(dim4(2,2), p1);

    const int start = 0;
    const double step = 1;
    const int d0 = 0;
    array interpolated = approx2(input,
                                 pos0, d0,
                                 pos1, d0 + 1,
                                 start, step,
                                 start, step);

    const float expected_interp[4] = {10, 30, 70, 90};
    array interpolated_gold(dim4(2,2), expected_interp);
    ASSERT_ARRAYS_NEAR(interpolated, interpolated_gold, 1e-5);
}

TEST(Approx2, CPPUniformFlippedDims)
{
    float inv[9] = {10, 20, 30,
                    40, 50, 60,
                    70, 80, 90};
    array input(dim4(3,3), inv);

    // generate grid of interpolation locations
    float p0[4] = {0, 2, 0, 2};
    float p1[4] = {0, 0, 2, 2};
    array pos0(dim4(2,2), p0);
    array pos1(dim4(2,2), p1);

    const int start = 0;
    const double step = 1;
    const int d1 = 1;
    array interpolated = approx2(input,
                                 pos1, d1,
                                 pos0, d1 - 1,
                                 start, step,
                                 start, step);

    const float expected_interp[4] = {10, 70, 30, 90};
    array interpolated_gold(dim4(2,2), expected_interp);
    ASSERT_ARRAYS_NEAR(interpolated, interpolated_gold, 1e-5);
}

TEST(Approx2, OtherDimLinear)
{
    int start = 0;
    int stop = 10000;
    int step = 100;
    int num = 1000;
    array xi = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array zi = 4 * xi * yi - 3 * xi;
    array xo = af::round(step * randu(num, 2, 2, 2));
    array yo = af::round(step * randu(num, 2, 2, 2));
    array zo = 4 * xo * yo - 3 * xo;
    for (int d = 1; d < 3; d++) {
        dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        array zi_reordered = reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered = reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array zo_reordered = approx2(zi_reordered,
                                     xo_reordered, d,
                                     yo_reordered, d + 1,
                                     start, step, start, step,
                                     AF_INTERP_LINEAR);
        rdims[d] = 0;
        rdims[0] = d;
        array res = af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx2, OtherDimCubic)
{
    float start = 0;
    float stop = 100;
    float step = 0.01;
    int num = 1000;
    array xi = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array zi = 4 * sin(xi) * cos(yi);
    array xo = af::round(step * randu(num, 2, 2, 2));
    array yo = af::round(step * randu(num, 2, 2, 2));
    array zo = 4 * sin(xo) * cos(yo);
    for (int d = 1; d < 3; d++) {
        dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        array zi_reordered = reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered = reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array zo_reordered = approx2(zi_reordered,
                                     xo_reordered, d,
                                     yo_reordered, d + 1,
                                     start, step, start, step,
                                     AF_INTERP_CUBIC);
        rdims[d] = 0;
        rdims[0] = d;
        array res = reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}
