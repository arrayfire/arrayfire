/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/exception.h>
#include <af/gfor.h>
#include <af/random.h>
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
class Approx2 : public ::testing::Test {
   public:
    virtual void SetUp() {
        SUPPORTED_TYPE_CHECK(T);
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// register the type list
TYPED_TEST_SUITE(Approx2, TestTypes);

template<typename T>
void approx2Test(string pTestFile, const unsigned resultIdx,
                 const af_interp_type method, bool isSubRef = false,
                 const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<T>> tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray   = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(),
                                   qdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));

    ASSERT_SUCCESS(
        af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

    if (inArray != 0) af_release_array(inArray);
    if (pos0Array != 0) af_release_array(pos0Array);
    if (pos1Array != 0) af_release_array(pos1Array);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

TYPED_TEST(Approx2, Approx2Nearest) {
    approx2Test<TypeParam>(string(TEST_DIR "/approx/approx2.test"), 0,
                           AF_INTERP_NEAREST);
}

TYPED_TEST(Approx2, Approx2Linear) {
    approx2Test<TypeParam>(string(TEST_DIR "/approx/approx2.test"), 1,
                           AF_INTERP_LINEAR);
}
TYPED_TEST(Approx2, NearestBatch) {
    approx2Test<TypeParam>(string(TEST_DIR "/approx/approx2_batch.test"), 0,
                           AF_INTERP_NEAREST);
}
TYPED_TEST(Approx2, LinearBatch) {
    approx2Test<TypeParam>(string(TEST_DIR "/approx/approx2_batch.test"), 1,
                           AF_INTERP_LINEAR);
}

// Test Argument Failure Cases
template<typename T>
void approx2ArgsTest(string pTestFile, const af_interp_type method,
                     const af_err err) {
    SUPPORTED_TYPE_CHECK(T);
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<T>> tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray   = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray  = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(),
                                   qdims.get(),
                                   (af_dtype)dtype_traits<BT>::af_type));

    ASSERT_EQ(err,
              af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));

    if (inArray != 0) af_release_array(inArray);
    if (pos0Array != 0) af_release_array(pos0Array);
    if (pos1Array != 0) af_release_array(pos1Array);
    if (outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Approx2, Approx2NearestArgsPos3D) {
    approx2ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx2_pos3d.test"),
                               AF_INTERP_NEAREST, AF_ERR_SIZE);
}

TYPED_TEST(Approx2, Approx2LinearArgsPos3D) {
    approx2ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx2_pos3d.test"),
                               AF_INTERP_LINEAR, AF_ERR_SIZE);
}

TYPED_TEST(Approx2, Approx2NearestArgsPosUnequal) {
    approx2ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx2_unequal.test"),
                               AF_INTERP_NEAREST, AF_ERR_SIZE);
}

template<typename T>
void approx2ArgsTestPrecision(string pTestFile, const unsigned resultIdx,
                              const af_interp_type method) {
    UNUSED(resultIdx);
    SUPPORTED_TYPE_CHECK(T);
    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    af_array inArray   = 0;
    af_array pos0Array = 0;
    af_array pos1Array = 0;
    af_array outArray  = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(af_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&pos0Array, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&pos1Array, &(in[2].front()), qdims.ndims(),
                                   qdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if ((af_dtype)dtype_traits<T>::af_type == c32 ||
        (af_dtype)dtype_traits<T>::af_type == c64) {
        ASSERT_EQ(AF_ERR_ARG, af_approx2(&outArray, inArray, pos0Array,
                                         pos1Array, method, 0));
    } else {
        ASSERT_SUCCESS(
            af_approx2(&outArray, inArray, pos0Array, pos1Array, method, 0));
    }

    if (inArray != 0) af_release_array(inArray);
    if (pos0Array != 0) af_release_array(pos0Array);
    if (pos1Array != 0) af_release_array(pos1Array);
    if (outArray != 0) af_release_array(outArray);
}

#define APPROX2_ARGSP(desc, file, resultIdx, method)                       \
    TYPED_TEST(Approx2, desc) {                                            \
        approx2ArgsTestPrecision<TypeParam>(                               \
            string(TEST_DIR "/approx/" #file ".test"), resultIdx, method); \
    }

APPROX2_ARGSP(Approx2NearestArgsPrecision, approx2, 0, AF_INTERP_NEAREST);
APPROX2_ARGSP(Approx2LinearArgsPrecision, approx2, 1, AF_INTERP_LINEAR);

//////////////////////////////////// CPP ////////////////////////////////////
//
TEST(Approx2, CPP) {
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<float>> tests;
    readTests<BT, float, float>(string(TEST_DIR "/approx/approx2.test"),
                                numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims, &(in[0].front()));
    array pos0(pdims, &(in[1].front()));
    array pos1(qdims, &(in[2].front()));
    array output = approx2(input, pos0, pos1, AF_INTERP_LINEAR, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2Cubic, CPP) {
    const unsigned resultIdx = 0;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<float>> tests;
    readTests<BT, float, float>(string(TEST_DIR "/approx/approx2_cubic.test"),
                                numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];
    dim4 qdims = numDims[2];

    array input(idims, &(in[0].front()));
    input = input.T();
    array pos0(pdims, &(in[1].front()));
    array pos1(qdims, &(in[2].front()));
    pos0         = tile(pos0, 1, pos0.dims(0));
    pos1         = tile(pos1.T(), pos1.dims(0));
    array output = approx2(input, pos0, pos1, AF_INTERP_BICUBIC_SPLINE, 0).T();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;

    float max = real(outData[0]), min = real(outData[0]);
    for (int i = 1; i < (int)nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) <
               0.01 * range);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx2, CPPNearestBatch) {
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) =
            approx2(input(span, span, i), pos(span, span, i),
                    qos(span, span, i), AF_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) =
            approx2(input(span, span, i), pos(span, span, i),
                    qos(span, span, i), AF_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx2, CPPLinearBatch) {
    array input = randu(200, 100, 10);
    array pos   = input.dims(0) * randu(100, 100, 10);
    array qos   = input.dims(1) * randu(100, 100, 10);

    array outBatch = approx2(input, pos, qos, AF_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(2); i++) {
        outSerial(span, span, i) =
            approx2(input(span, span, i), pos(span, span, i),
                    qos(span, span, i), AF_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(2)) {
        outGFOR(span, span, i) =
            approx2(input(span, span, i), pos(span, span, i),
                    qos(span, span, i), AF_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx2, CPPNearestMaxDims) {
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

TEST(Approx2, CPPLinearMaxDims) {
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

TEST(Approx2, CPPCubicMaxDims) {
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

TEST(Approx2, OtherDimLinear) {
    int start = 0;
    int stop  = 10000;
    int step  = 100;
    int num   = 1000;
    array xi  = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi  = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array zi  = 4 * xi * yi - 3 * xi;
    array xo  = af::round(step * randu(num, 2, 2, 2));
    array yo  = af::round(step * randu(num, 2, 2, 2));
    array zo  = 4 * xo * yo - 3 * xo;
    for (int d = 1; d < 3; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array zi_reordered =
            reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered =
            reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array zo_reordered =
            approx2(zi_reordered, xo_reordered, d, start, step, yo_reordered,
                    d + 1, start, step, AF_INTERP_LINEAR);
        rdims[d] = 0;
        rdims[0] = d;
        array res =
            af::reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx2, OtherDimCubic) {
    float start = 0;
    float stop  = 100;
    float step  = 0.01;
    int num     = 1000;
    array xi    = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi    = af::tile(seq(start, stop, step), 1, 2, 2, 2);
    array zi    = 4 * sin(xi) * cos(yi);
    array xo    = af::round(step * randu(num, 2, 2, 2));
    array yo    = af::round(step * randu(num, 2, 2, 2));
    array zo    = 4 * sin(xo) * cos(yo);
    for (int d = 1; d < 3; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array zi_reordered =
            reorder(zi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered =
            reorder(yo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array zo_reordered =
            approx2(zi_reordered, xo_reordered, d, start, step, yo_reordered,
                    d + 1, start, step, AF_INTERP_CUBIC);
        rdims[d] = 0;
        rdims[0] = d;
        array res =
            reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, af::max<float>(af::abs(res - yo)), 1E-3);
    }
}

TEST(Approx2, CPPUsage) {
    //! [ex_signal_approx2]

    // Input data array.
    float input_vals[9] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    array input(3, 3, input_vals);
    // [3 3 1 1]
    //     1.0000     2.0000     3.0000
    //     1.0000     2.0000     3.0000
    //     1.0000     2.0000     3.0000

    // First array of positions to be found along the first dimension.
    float pv0[4] = {0.5, 1.5, 0.5, 1.5};
    array pos0(2, 2, pv0);
    // [2 2 1 1]
    //     0.5000     0.5000
    //     1.5000     1.5000

    // Second array of positions to be found along the second
    // dimension.
    float pv1[4] = {0.5, 0.5, 1.5, 1.5};
    array pos1(2, 2, pv1);
    // [2 2 1 1]
    //     0.5000     1.5000
    //     0.5000     1.5000

    array interp = approx2(input, pos0, pos1);
    // [2 2 1 1]
    //     1.5000     2.5000
    //     1.5000     2.5000

    //! [ex_signal_approx2]

    float expected_interp[4] = {1.5, 1.5, 2.5, 2.5};

    array interp_gold(2, 2, expected_interp);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx2, CPPUniformUsage) {
    //! [ex_signal_approx2_uniform]

    // Input data array.
    float input_vals[9] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    array input(3, 3, input_vals);
    // [3 3 1 1]
    //     1.0000     2.0000     3.0000
    //     1.0000     2.0000     3.0000
    //     1.0000     2.0000     3.0000

    // First array of positions to be found along the interpolation
    // dimension, `interp_dim0`.
    float pv0[4] = {0.5, 1.5, 0.5, 1.5};
    array pos0(2, 2, pv0);
    // [2 2 1 1]
    //     0.5000     0.5000
    //     1.5000     1.5000

    // Second array of positions to be found along the interpolation
    // dimension, `interp_dim1`.
    float pv1[4] = {0.5, 0.5, 1.5, 1.5};
    array pos1(2, 2, pv1);
    // [2 2 1 1]
    //     0.5000     1.5000
    //     0.5000     1.5000

    // Define range of indices with which the input values will
    // correspond along both dimensions to be interpolated.
    const double idx_start_dim0 = 0.0;
    const double idx_step_dim0  = 1.0;
    const int interp_dim0       = 0;
    const int interp_dim1       = 1;
    array interp =
        approx2(input, pos0, interp_dim0, idx_start_dim0, idx_step_dim0, pos1,
                interp_dim1, idx_start_dim0, idx_step_dim0);
    // [2 2 1 1]
    //     1.5000     2.5000
    //     1.5000     2.5000

    //! [ex_signal_approx2_uniform]

    float expected_interp[4] = {1.5, 1.5, 2.5, 2.5};

    array interp_gold(2, 2, expected_interp);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx2, CPPUniformOneDimIndices) {
    float inv[9] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    array input(dim4(3, 3), inv);

    float p0[3] = {0.0, 1.0, 2.0};
    float p1[3] = {0.0, 1.0, 2.0};
    array pos0(dim4(3, 1), p0);
    array pos1(dim4(3, 1), p1);

    const int pos0_interp_grid_start   = 0;
    const double pos0_interp_grid_step = 1;
    array interpolated =
        approx2(input, pos0, 0, pos0_interp_grid_start, pos0_interp_grid_step,
                pos1, 1, pos0_interp_grid_start, pos0_interp_grid_step);

    float expected_interp[3] = {10.0, 50.0, 90.0};

    array interpolated_gold(dim4(3, 1), expected_interp);
    ASSERT_ARRAYS_EQ(interpolated, interpolated_gold);
}

TEST(Approx2, CPPUniformTwoDimIndices) {
    float inv[9] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    array input(dim4(3, 3), inv);

    float p0[4] = {0, 2, 0, 2};
    float p1[4] = {0, 0, 2, 2};
    array pos0(dim4(2, 2), p0);
    array pos1(dim4(2, 2), p1);
    const int pos0_interp_grid_start   = 0;
    const double pos0_interp_grid_step = 1;
    const int pos0_interp_dim          = 0;
    const int pos1_interp_dim          = 1;

    array interpolated =
        approx2(input, pos0, pos0_interp_dim, pos0_interp_grid_start,
                pos0_interp_grid_step, pos1, pos1_interp_dim,
                pos0_interp_grid_start, pos0_interp_grid_step);

    float expected_interp[4] = {10.0, 30.0, 70.0, 90.0};
    array interpolated_gold(dim4(2, 2), expected_interp);
    ASSERT_ARRAYS_EQ(interpolated, interpolated_gold);
}

TEST(Approx2, CPPUniformInvalidStepSize) {
    try {
        float inv[9] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
        array in(dim4(3, 3), inv);
        float pv[3] = {0.0, -1.0, -2.0};
        array pos(dim4(3, 1), pv);
        const int pos0_interp_grid_start   = -1;
        const double pos0_interp_grid_step = 0;
        const int pos0_interp_dim          = 0;
        const int pos1_interp_dim          = 1;

        array interpolated =
            approx2(in, pos, pos0_interp_dim, pos0_interp_grid_start,
                    pos0_interp_grid_step, pos, pos1_interp_dim,
                    pos0_interp_grid_start, pos0_interp_grid_step);
        FAIL() << "Expected af::exception\n";
    } catch (af::exception& ex) { SUCCEED(); } catch (...) {
        FAIL() << "Expected af::exception\n";
    }
}

TEST(Approx2, CPPUniformColumnMajorInterpolation) {
    float inv[9] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    array input(dim4(3, 3), inv);

    float p0[4] = {0, 2, 0, 2};
    float p1[4] = {0, 0, 2, 2};
    array pos0(dim4(2, 2), p0);
    array pos1(dim4(2, 2), p1);
    const int pos0_interp_dim          = 0;
    const int pos1_interp_dim          = 1;
    const int pos0_interp_grid_start   = 0;
    const double pos0_interp_grid_step = 1;

    array first = approx2(input, pos0, pos0_interp_dim, pos0_interp_grid_start,
                          pos0_interp_grid_step, pos1, pos1_interp_dim,
                          pos0_interp_grid_start, pos0_interp_grid_step);

    array second = approx2(input, pos1, pos1_interp_dim, pos0_interp_grid_start,
                           pos0_interp_grid_step, pos0, pos0_interp_dim,
                           pos0_interp_grid_start, pos0_interp_grid_step);

    // Verify.
    float expected_interp[4] = {10.0, 30.0, 70.0, 90.0};
    array interpolated_gold(dim4(2, 2), expected_interp);
    ASSERT_ARRAYS_EQ(first, interpolated_gold);
    ASSERT_ARRAYS_EQ(first, second);
}

TEST(Approx2, CPPUniformRowMajorInterpolation) {
    float inv[9] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    array input(dim4(3, 3), inv);

    float p0[4] = {0, 2, 0, 2};
    float p1[4] = {0, 0, 2, 2};
    array pos0(dim4(2, 2), p0);
    array pos1(dim4(2, 2), p1);
    const int pos0_interp_grid_start   = 0;
    const double pos0_interp_grid_step = 1;

    array first =
        approx2(input, pos0, 1, pos0_interp_grid_start, pos0_interp_grid_step,
                pos1, 0, pos0_interp_grid_start, pos0_interp_grid_step);

    array second =
        approx2(input, pos1, 0, pos0_interp_grid_start, pos0_interp_grid_step,
                pos0, 1, pos0_interp_grid_start, pos0_interp_grid_step);

    // Verify.
    float expected_interp[4] = {10.0, 70.0, 30.0, 90.0};
    array interpolated_gold(dim4(2, 2), expected_interp);
    ASSERT_ARRAYS_EQ(first, interpolated_gold);
    ASSERT_ARRAYS_EQ(first, second);
}

TEST(Approx2, CPPEmptyPos) {
    float inv[3] = {10.0, 20.0, 30.0};
    array in(dim4(3, 1), inv);
    array pos;
    array interpolated = approx2(in, pos, pos);
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

TEST(Approx2, CPPEmptyInput) {
    array in;
    float pv[3] = {0.0, 1.0, 2.0};
    array pos(dim4(3, 1), pv);

    array interpolated = approx2(in, pos, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

TEST(Approx2, CPPEmptyPosAndInput) {
    array in;
    array pos;
    array interpolated = approx2(in, pos, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interpolated.isempty());
}

template<typename T>
class Approx2V2 : public ::testing::Test {
   protected:
    typedef typename dtype_traits<T>::base_type BT;

    vector<T> h_gold_cast;
    vector<T> h_in_cast;
    vector<BT> h_pos1_cast;
    vector<BT> h_pos2_cast;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos1_dims;
    dim4 pos2_dims;

    af_array gold;
    af_array in;
    af_array pos1;
    af_array pos2;

    Approx2V2() : gold(0), in(0), pos1(0), pos2(0) {}

    void SetUp() {}

    void releaseArrays() {
        if (pos2 != 0) { ASSERT_SUCCESS(af_release_array(pos2)); }
        if (pos1 != 0) { ASSERT_SUCCESS(af_release_array(pos1)); }
        if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
        if (gold != 0) { ASSERT_SUCCESS(af_release_array(gold)); }
    }

    void TearDown() { releaseArrays(); }

    void setTestData(float* h_gold, dim4 gold_dims, float* h_in, dim4 in_dims,
                     float* h_pos1, dim4 pos1_dims, float* h_pos2,
                     dim4 pos2_dims) {
        releaseArrays();

        gold = 0;
        in   = 0;
        pos1 = 0;
        pos2 = 0;

        this->gold_dims = gold_dims;
        this->in_dims   = in_dims;
        this->pos1_dims = pos1_dims;
        this->pos2_dims = pos2_dims;

        for (int i = 0; i < gold_dims.elements(); ++i) {
            h_gold_cast.push_back(static_cast<T>(h_gold[i]));
        }
        for (int i = 0; i < in_dims.elements(); ++i) {
            h_in_cast.push_back(static_cast<T>(h_in[i]));
        }
        for (int i = 0; i < pos1_dims.elements(); ++i) {
            h_pos1_cast.push_back(static_cast<BT>(h_pos1[i]));
        }
        for (int i = 0; i < pos2_dims.elements(); ++i) {
            h_pos2_cast.push_back(static_cast<BT>(h_pos2[i]));
        }

        ASSERT_SUCCESS(af_create_array(&gold, &h_gold_cast.front(),
                                       gold_dims.ndims(), gold_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&in, &h_in_cast.front(), in_dims.ndims(),
                                       in_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&pos1, &h_pos1_cast.front(),
                                       pos1_dims.ndims(), pos1_dims.get(),
                                       (af_dtype)dtype_traits<BT>::af_type));
        ASSERT_SUCCESS(af_create_array(&pos2, &h_pos2_cast.front(),
                                       pos2_dims.ndims(), pos2_dims.get(),
                                       (af_dtype)dtype_traits<BT>::af_type));
    }

    void testSpclOutArray(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        af_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (af_dtype)dtype_traits<T>::af_type, &metadata);

        ASSERT_SUCCESS(
            af_approx2_v2(&out, in, pos1, pos2, AF_INTERP_LINEAR, 0));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }

    void testSpclOutArrayUniform(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        af_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (af_dtype)dtype_traits<T>::af_type, &metadata);

        ASSERT_SUCCESS(af_approx2_uniform_v2(&out, in, pos1, 0, 0.0, 1.0, pos2,
                                             1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }
};

TYPED_TEST_SUITE(Approx2V2, TestTypes);

class SimpleTestData {
   public:
    static const int h_gold_size = 4;
    static const int h_in_size   = 9;
    static const int h_pos1_size = 4;
    static const int h_pos2_size = 4;

    vector<float> h_gold;
    vector<float> h_in;
    vector<float> h_pos1;
    vector<float> h_pos2;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos1_dims;
    dim4 pos2_dims;

   public:
    SimpleTestData()
        : gold_dims(2, 2), in_dims(3, 3), pos1_dims(2, 2), pos2_dims(2, 2) {
        float gold_arr[h_gold_size] = {1.5, 1.5, 2.5, 2.5};

        float in_arr[h_in_size] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};

        float pos1_arr[h_pos1_size] = {0.5, 1.5, 0.5, 1.5};

        float pos2_arr[h_pos2_size] = {0.5, 0.5, 1.5, 1.5};

        h_gold.assign(gold_arr, gold_arr + h_gold_size);
        h_in.assign(in_arr, in_arr + h_in_size);
        h_pos1.assign(pos1_arr, pos1_arr + h_pos1_size);
        h_pos2.assign(pos2_arr, pos2_arr + h_pos2_size);
    }
};

template<typename T>
class Approx2V2Simple : public Approx2V2<T> {
   protected:
    void SetUp() {
        SUPPORTED_TYPE_CHECK(T);
        SimpleTestData data;
        this->setTestData(&data.h_gold.front(), data.gold_dims,
                          &data.h_in.front(), data.in_dims,
                          &data.h_pos1.front(), data.pos1_dims,
                          &data.h_pos2.front(), data.pos2_dims);
    }
};

TYPED_TEST_SUITE(Approx2V2Simple, TestTypes);

TYPED_TEST(Approx2V2Simple, UseNullOutputArray) {
    this->testSpclOutArray(NULL_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UseFullExistingOutputArray) {
    this->testSpclOutArray(FULL_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UseExistingOutputSubArray) {
    this->testSpclOutArray(SUB_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UseReorderedOutputArray) {
    this->testSpclOutArray(REORDERED_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UniformUseNullOutputArray) {
    this->testSpclOutArrayUniform(NULL_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UniformUseFullExistingOutputArray) {
    this->testSpclOutArrayUniform(FULL_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UniformUseExistingOutputSubArray) {
    this->testSpclOutArrayUniform(SUB_ARRAY);
}

TYPED_TEST(Approx2V2Simple, UniformUseReorderedOutputArray) {
    this->testSpclOutArrayUniform(REORDERED_ARRAY);
}

class Approx2NullArgs : public ::testing::Test {
   protected:
    af_array out;
    af_array in;
    af_array pos1;
    af_array pos2;

    Approx2NullArgs() : out(0), in(0), pos1(0), pos2(0) {}

    void SetUp() {
        SimpleTestData data;
        ASSERT_SUCCESS(af_create_array(&in, &data.h_in.front(),
                                       data.in_dims.ndims(), data.in_dims.get(),
                                       f32));
        ASSERT_SUCCESS(af_create_array(&pos1, &data.h_pos1.front(),
                                       data.pos1_dims.ndims(),
                                       data.pos1_dims.get(), f32));
        ASSERT_SUCCESS(af_create_array(&pos2, &data.h_pos2.front(),
                                       data.pos2_dims.ndims(),
                                       data.pos2_dims.get(), f32));
    }

    void TearDown() {
        if (pos2 != 0) { ASSERT_SUCCESS(af_release_array(pos2)); }
        if (pos1 != 0) { ASSERT_SUCCESS(af_release_array(pos1)); }
        if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
    }
};

TEST_F(Approx2NullArgs, NullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG, af_approx2(out_ptr, this->in, this->pos1, this->pos2,
                                     AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2(&this->out, 0, this->pos1, this->pos2,
                                     AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, NullPos1Array) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2(&this->out, this->in, 0, this->pos2,
                                     AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, NullPos2Array) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2(&this->out, this->in, this->pos1, 0,
                                     AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, V2NullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG, af_approx2_v2(out_ptr, this->in, this->pos1,
                                        this->pos2, AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, V2NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2_v2(&this->out, 0, this->pos1, this->pos2,
                                        AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, V2NullPos1Array) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2_v2(&this->out, this->in, 0, this->pos2,
                                        AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, V2NullPos2Array) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2_v2(&this->out, this->in, this->pos1, 0,
                                        AF_INTERP_LINEAR, 0.f));
}

TEST_F(Approx2NullArgs, UniformNullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_approx2_uniform(out_ptr, this->in, this->pos1, 0, 0.0, 1.0,
                                 this->pos2, 1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, UniformNullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx2_uniform(&this->out, 0, this->pos1, 0, 0.0, 1.0,
                                 this->pos2, 1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, UniformNullPos1Array) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx2_uniform(&this->out, this->in, 0, 0, 0.0, 1.0,
                                 this->pos2, 1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, UniformNullPos2Array) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx2_uniform(&this->out, this->in, this->pos1, 0, 0.0, 1.0,
                                 0, 1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, V2UniformNullOutputPtr) {
    af_array* out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG, af_approx2_uniform_v2(out_ptr, this->in, this->pos1,
                                                0, 0.0, 1.0, this->pos2, 1, 0.0,
                                                1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, V2UniformNullInputArray) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2_uniform_v2(&this->out, 0, this->pos1, 0,
                                                0.0, 1.0, this->pos2, 1, 0.0,
                                                1.0, AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, V2UniformNullPos1Array) {
    ASSERT_EQ(AF_ERR_ARG, af_approx2_uniform_v2(&this->out, this->in, 0, 0, 0.0,
                                                1.0, this->pos2, 1, 0.0, 1.0,
                                                AF_INTERP_LINEAR, 0));
}

TEST_F(Approx2NullArgs, V2UniformNullPos2Array) {
    ASSERT_EQ(AF_ERR_ARG,
              af_approx2_uniform_v2(&this->out, this->in, this->pos1, 0, 0.0,
                                    1.0, 0, 1, 0.0, 1.0, AF_INTERP_LINEAR, 0));
}
