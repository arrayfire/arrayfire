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
    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
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
    typedef typename af::dtype_traits<T>::base_type BT;
    vector<af::dim4> numDims;
    vector<vector<BT> > in;
    vector<vector<T> > tests;
    readTests<BT, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<BT>::af_type));

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
    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::dim4 pdims = numDims[1];
    af::dim4 qdims = numDims[2];

    af_array inArray = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(input.front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(), pdims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(), qdims.get(), (af_dtype) af::dtype_traits<T>::af_type));


    if((af_dtype) af::dtype_traits<T>::af_type == c32 ||
       (af_dtype) af::dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
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
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
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
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t" << outData[elIter] << "at: " << elIter << std::endl;
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

    af::array input = af::randu(1, largeDim);
    af::array pos   = input.dims(0) * af::randu(1, 10);
    af::array qos   = input.dims(1) * af::randu(1, 10);
    af::array out   = af::approx2(input, pos, qos, AF_INTERP_NEAREST);

    input = af::randu(1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_NEAREST);

    input = af::randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_NEAREST);

    SUCCEED();
}

TEST(Approx2, CPPLinearMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;

    af::array input = af::randu(1, largeDim);
    af::array pos   = input.dims(0) * af::randu(1, 10);
    af::array qos   = input.dims(1) * af::randu(1, 10);
    af::array out   = af::approx2(input, pos, qos, AF_INTERP_LINEAR);

    input = af::randu(1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_LINEAR);

    input = af::randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_LINEAR);

    SUCCEED();
}

TEST(Approx2, CPPCubicMaxDims)
{
    if (noDoubleTests<float>()) return;

    const size_t largeDim = 65535 * 32 + 1;

    af::array input = af::randu(1, largeDim);
    af::array pos   = input.dims(0) * af::randu(1, 10);
    af::array qos   = input.dims(1) * af::randu(1, 10);
    af::array out   = af::approx2(input, pos, qos, AF_INTERP_BICUBIC);

    input = af::randu(1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_BICUBIC);

    input = af::randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * af::randu(1, 1, 1, largeDim);
    qos   = input.dims(1) * af::randu(1, 1, 1, largeDim);
    out   = af::approx2(input, pos, qos, AF_INTERP_BICUBIC);

    SUCCEED();
}

TEST(Approx2, OtherDimLinear)
{
    int start = 0;
    int stop = 10000;
    int step = 100;
    int num = 1000;
    af::array xi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array yi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array zi = 4 * xi * yi - 3 * xi;
    af::array xo = af::round(step * af::randu(num, 2, 2, 2));
    af::array yo = af::round(step * af::randu(num, 2, 2, 2));
    af::array zo = 4 * xo * yo - 3 * xo;
    for (int d = 1; d < 3; d++) {
        af::dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        af::array zi_reordered = af::reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array xo_reordered = af::reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array yo_reordered = af::reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array zo_reordered = af::approx2(zi_reordered,
                                             xo_reordered, d,
                                             yo_reordered, d + 1,
                                             start, step, start, step,
                                             AF_INTERP_LINEAR);
        rdims[d] = 0;
        rdims[0] = d;
        af::array res = af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx2, OtherDimCubic)
{
    float start = 0;
    float stop = 100;
    float step = 0.01;
    int num = 1000;
    af::array xi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array yi = af::tile(af::seq(start, stop, step), 1, 2, 2, 2);
    af::array zi = 4 * sin(xi) * cos(yi);
    af::array xo = af::round(step * af::randu(num, 2, 2, 2));
    af::array yo = af::round(step * af::randu(num, 2, 2, 2));
    af::array zo = 4 * sin(xo) * cos(yo);
    for (int d = 1; d < 3; d++) {
        af::dim4 rdims(0,1,2,3);
        rdims[0] = d;
        rdims[d] = 0;

        af::array zi_reordered = af::reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array xo_reordered = af::reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array yo_reordered = af::reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        af::array zo_reordered = af::approx2(zi_reordered,
                                             xo_reordered, d,
                                             yo_reordered, d + 1,
                                             start, step, start, step,
                                             AF_INTERP_CUBIC);
        rdims[d] = 0;
        rdims[0] = d;
        af::array res = af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}
