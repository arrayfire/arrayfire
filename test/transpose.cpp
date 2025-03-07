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
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Transpose : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat2D.push_back(af_make_seq(2, 7, 1));
        subMat2D.push_back(af_make_seq(2, 7, 1));

        subMat3D.push_back(af_make_seq(2, 7, 1));
        subMat3D.push_back(af_make_seq(2, 7, 1));
        subMat3D.push_back(af_span);
    }
    vector<af_seq> subMat2D;
    vector<af_seq> subMat3D;
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, uchar,
                         short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Transpose, TestTypes);

template<typename T>
void trsTest(string pTestFile, bool isSubRef = false,
             const vector<af_seq> *seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    af_array outArray = 0;
    af_array inArray  = 0;
    T *outData;
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    // check if the test is for indexed Array
    if (isSubRef) {
        dim4 newDims(dims[1] - 4, dims[0] - 4, dims[2], dims[3]);
        af_array subArray = 0;
        ASSERT_SUCCESS(
            af_index(&subArray, inArray, seqv->size(), &seqv->front()));
        ASSERT_SUCCESS(af_transpose(&outArray, subArray, false));
        // destroy the temporary indexed Array
        ASSERT_SUCCESS(af_release_array(subArray));

        dim_t nElems;
        ASSERT_SUCCESS(af_get_elements(&nElems, outArray));
        outData = new T[nElems];
    } else {
        ASSERT_SUCCESS(af_transpose(&outArray, inArray, false));
        outData = new T[dims.elements()];
    }

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData, outArray));

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(Transpose, Vector) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/vector.test"));
}

TYPED_TEST(Transpose, VectorBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/vector_batch.test"));
}

TYPED_TEST(Transpose, Square) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square.test"));
}

TYPED_TEST(Transpose, Rectangle) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle.test"));
}

TYPED_TEST(Transpose, Rectangle2) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle2.test"));
}

TYPED_TEST(Transpose, SquareBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square_batch.test"));
}

TYPED_TEST(Transpose, RectangleBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle_batch.test"));
}

TYPED_TEST(Transpose, RectangleBatch2) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

TYPED_TEST(Transpose, Square512x512) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/square2.test"));
}

TYPED_TEST(Transpose, SubRef) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/offset.test"), true,
                       &(this->subMat2D));
}

TYPED_TEST(Transpose, SubRefBatch) {
    trsTest<TypeParam>(string(TEST_DIR "/transpose/offset_batch.test"), true,
                       &(this->subMat3D));
}

////////////////////////////////////// CPP //////////////////////////////////
//
template<typename T>
void trsCPPTest(string pFileName) {
    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pFileName, numDims, in, tests);
    dim4 dims = numDims[0];

    SUPPORTED_TYPE_CHECK(T);

    array input(dims, &(in[0].front()));
    array output = transpose(input);

    T *outData = new T[dims.elements()];
    output.host((void *)outData);

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // cleanup
    delete[] outData;
}

TEST(Transpose, CPP_f64) {
    trsCPPTest<double>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

TEST(Transpose, CPP_f32) {
    trsCPPTest<float>(string(TEST_DIR "/transpose/rectangle_batch2.test"));
}

template<typename T>
void trsCPPConjTest(dim_t d0, dim_t d1 = 1, dim_t d2 = 1, dim_t d3 = 1) {
    vector<dim4> numDims;

    dim4 dims(d0, d1, d2, d3);

    SUPPORTED_TYPE_CHECK(T);

    array input    = randu(dims, (af_dtype)dtype_traits<T>::af_type);
    array output_t = transpose(input, false);
    array output_c = transpose(input, true);

    T *tData = new T[dims.elements()];
    T *cData = new T[dims.elements()];
    output_t.host((void *)tData);
    output_c.host((void *)cData);

    size_t nElems = dims.elements();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(real(tData[elIter]), real(cData[elIter]), 1e-6)
            << "at: " << elIter << endl;
        ASSERT_NEAR(-imag(tData[elIter]), imag(cData[elIter]), 1e-6)
            << "at: " << elIter << endl;
    }

    // cleanup
    delete[] tData;
    delete[] cData;
}

TEST(Transpose, CPP_c32_CONJ40x40) { trsCPPConjTest<cfloat>(40, 40); }

TEST(Transpose, CPP_c32_CONJ2000x1) { trsCPPConjTest<cfloat>(2000); }

TEST(Transpose, CPP_c32_CONJ20x20x5) { trsCPPConjTest<cfloat>(20, 20, 5); }

TEST(Transpose, MaxDim) {
    const size_t largeDim = 65535 * 33 + 1;

    array input  = range(dim4(2, largeDim, 1, 1));
    array gold   = range(dim4(largeDim, 2, 1, 1), 1);
    array output = transpose(input);

    ASSERT_EQ(output.dims(0), (int)largeDim);
    ASSERT_EQ(output.dims(1), 2);
    ASSERT_ARRAYS_EQ(gold, output);

    input  = range(dim4(2, 5, 1, largeDim));
    gold   = range(dim4(5, 2, 1, largeDim), 1);
    output = transpose(input);

    ASSERT_ARRAYS_EQ(gold, output);
}

TEST(Transpose, GFOR) {
    using af::constant;
    using af::max;
    using af::seq;
    using af::span;

    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 100, 3);

    gfor(seq ii, 3) { B(span, span, ii) = A(span, span, ii).T(); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = A(span, span, ii).T();
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(Transpose, SNIPPET_blas_func_transpose) {
    // clang-format off
    //! [ex_blas_func_transpose]
    //!
    // Create a, a 2x3 array
    array a = iota(dim4(2, 3));    // a = [0, 2, 4
                                   //      1, 3, 5]

    // Create b, the transpose of a
    array b = transpose(a);        // b = [0, 1,
                                   //      2, 3,
                                   //      4, 5]

    //! [ex_blas_func_transpose]
    // clang-format on

    using std::vector;
    vector<float> gold_b{0, 2, 4, 1, 3, 5};

    ASSERT_VEC_ARRAY_EQ(gold_b, b.dims(), b);
}
