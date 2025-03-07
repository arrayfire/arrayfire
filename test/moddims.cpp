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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <cstdlib>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Moddims : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat.push_back(af_make_seq(1, 2, 1));
        subMat.push_back(af_make_seq(1, 3, 1));
    }
    vector<af_seq> subMat;
};

// create a list of types to be tested
// TODO: complex types tests have to be added
typedef ::testing::Types<float, double, int, unsigned, char, unsigned char,
                         short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Moddims, TestTypes);

template<typename T>
void moddimsTest(string pTestFile, bool isSubRef = false,
                 const vector<af_seq> *seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    T *outData;

    if (isSubRef) {
        af_array inArray  = 0;
        af_array subArray = 0;
        af_array outArray = 0;

        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&subArray, inArray, seqv->size(), &seqv->front()));

        dim4 newDims(1);
        newDims[0] = 2;
        newDims[1] = 3;
        ASSERT_SUCCESS(
            af_moddims(&outArray, subArray, newDims.ndims(), newDims.get()));

        dim_t nElems;
        ASSERT_SUCCESS(af_get_elements(&nElems, outArray));

        outData = new T[nElems];
        ASSERT_SUCCESS(af_get_data_ptr((void *)outData, outArray));

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(subArray));
    } else {
        af_array inArray  = 0;
        af_array outArray = 0;

        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        dim4 newDims(1);
        newDims[0] = dims[1];
        newDims[1] = dims[0] * dims[2];
        ASSERT_SUCCESS(
            af_moddims(&outArray, inArray, newDims.ndims(), newDims.get()));

        outData = new T[dims.elements()];
        ASSERT_SUCCESS(af_get_data_ptr((void *)outData, outArray));

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
    }

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }
    delete[] outData;
}

TYPED_TEST(Moddims, Basic) {
    moddimsTest<TypeParam>(string(TEST_DIR "/moddims/basic.test"));
}

TYPED_TEST(Moddims, Subref) {
    moddimsTest<TypeParam>(string(TEST_DIR "/moddims/subref.test"), true,
                           &(this->subMat));
}

template<typename T>
void moddimsArgsTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array outArray2 = 0;
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    dim4 newDims(1);
    newDims[0] = dims[1];
    newDims[1] = dims[0] * dims[2];
    ASSERT_SUCCESS(af_moddims(&outArray, inArray, 0, newDims.get()));
    ASSERT_EQ(AF_ERR_ARG,
              af_moddims(&outArray2, inArray, newDims.ndims(), NULL));

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(Moddims, InvalidArgs) {
    moddimsArgsTest<TypeParam>(string(TEST_DIR "/moddims/basic.test"));
}

template<typename T>
void moddimsMismatchTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    af_array inArray  = 0;
    af_array outArray = 0;
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    dim4 newDims(1);
    newDims[0] = dims[1] - 1;
    newDims[1] = (dims[0] - 1) * dims[2];
    ASSERT_EQ(AF_ERR_SIZE,
              af_moddims(&outArray, inArray, newDims.ndims(), newDims.get()));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(Moddims, Mismatch) {
    moddimsMismatchTest<TypeParam>(string(TEST_DIR "/moddims/basic.test"));
}

/////////////////////////////////// CPP ///////////////////////////////////
//

using af::array;

template<typename T>
void cppModdimsTest(string pTestFile, bool isSubRef = false,
                    const vector<af_seq> *seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    T *outData;

    if (isSubRef) {
        array input(dims, &(in[0].front()));

        array subArray = input(seqv->at(0), seqv->at(1));

        dim4 newDims(1);
        newDims[0]   = 2;
        newDims[1]   = 3;
        array output = moddims(subArray, newDims.ndims(), newDims.get());

        dim_t nElems = output.elements();
        outData      = new T[nElems];
        output.host((void *)outData);
    } else {
        array input(dims, &(in[0].front()));

        dim4 newDims(1);
        newDims[0] = dims[1];
        newDims[1] = dims[0] * dims[2];

        array output = moddims(input, newDims.ndims(), newDims.get());

        outData = new T[dims.elements()];
        output.host((void *)outData);
    }

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }
    delete[] outData;
}

TEST(Moddims, Basic_CPP) {
    cppModdimsTest<float>(string(TEST_DIR "/moddims/basic.test"));
}

TEST(Moddims, Subref_CPP) {
    vector<af_seq> subMat;
    subMat.push_back(af_make_seq(1, 2, 1));
    subMat.push_back(af_make_seq(1, 3, 1));
    cppModdimsTest<float>(string(TEST_DIR "/moddims/subref.test"), true,
                          &subMat);
}

TEST(Moddims, jit) {
    using namespace af;
    array c1 = constant(1, 10, 5);
    c1.eval();
    array c2 = randu(10, 10);

    vector<float> hc2(100);
    c2.host(hc2.data());

    array c3 = c2(span, seq(5));
    c3.eval();

    array a = c1;
    a       = a + c3;
    a       = moddims(a, 5, 10);
    a       = a + constant(2, 5, 10);

    for (int i = 0; i < hc2.size(); i++) { hc2[i] += 3; }

    array gold(10, 5, hc2.data());
    gold = moddims(gold, 5, 10);
    ASSERT_ARRAYS_EQ(gold, a);
}

TEST(Moddims, JitNested) {
    array a    = af::constant(1, 5, 5);
    array b    = moddims(moddims(moddims(a, 25), 1, 5, 5), 5, 5);
    array gold = af::constant(1, 5, 5);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, b);
}

TEST(Moddims, JitDuplicate) {
    array a = af::constant(1, 5, 5);
    array b = af::moddims(a, 25);
    array c = b + b;

    array gold = af::constant(2, 25);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, c);
}

TEST(Moddims, JitNestedAndDuplicate) {
    array a = af::constant(1, 10, 10);
    array b = af::constant(1, 10, 10);
    array c = af::constant(2, 100) + moddims(a + b, 100);
    array d = moddims(
        moddims(af::constant(2, 1, 10, 10) + moddims(c, 1, 10, 10), 100), 10,
        10);
    array e    = d + d;
    array gold = af::constant(12, 10, 10);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, e);
}

TEST(Moddims, JitTileThenModdims) {
    array a    = af::constant(1, 10);
    array b    = tile(a, 1, 10);
    array c    = moddims(b, 100);
    array gold = af::constant(1, 100);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, c);
}

TEST(Moddims, JitModdimsThenTiled) {
    array a    = af::constant(1, 10);
    array b    = moddims(a, 1, 10);
    array c    = tile(b, 10);
    array gold = af::constant(1, 10, 10);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, c);
}

TEST(Moddims, JitTileThenMultipleModdims) {
    array a    = af::constant(1, 10);
    array b    = tile(a, 1, 10);
    array c    = moddims(moddims(b, 100), 10, 10);
    array gold = af::constant(1, 10, 10);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, c);
}

TEST(Moddims, JitMultipleModdimsThenTiled) {
    array a    = af::constant(1, 10);
    array b    = moddims(moddims(a, 1, 10), 1, 1, 10);
    array c    = tile(b, 10);
    array gold = af::constant(1, 10, 1, 10);
    gold.eval();
    ASSERT_ARRAYS_EQ(gold, c);
}

TEST(Moddims, SNIPPET_data_func_moddims) {
    // clang-format off
    //! [ex_data_func_moddims]
    //!
    // Create a, a 2x3 array
    array a = iota(dim4(2, 3));           // a = [0, 2, 4,
                                          //      1, 3, 5]

    // Create b by modifying the dimensions of a to the shape described by a dim4 object
    array b = moddims(a, dim4(3, 2));     // b = [0, 3,
                                          //      1, 4,
                                          //      2, 5]

    // Create c by modifying the dimensions of a to the shape described by dimension length parameters
    array c = moddims(a, 3, 2);           // c = [0, 3,
                                          //      1, 4,
                                          //      2, 5]

    // Create d by modifying the dimensions of a to the shape described by an array of ndims dimensions
    vector<dim_t> x{3, 2};
    array d = moddims(a, 2, x.data());    // d = [0, 3,
                                          //      1, 4,
                                          //      2, 5]

    //! [ex_data_func_moddims]
    // clang-format on

    vector<float> gold_a{0, 1, 2, 3, 4, 5};

    ASSERT_VEC_ARRAY_EQ(gold_a, dim4(3, 2), b);
    ASSERT_VEC_ARRAY_EQ(gold_a, dim4(3, 2), c);
    ASSERT_VEC_ARRAY_EQ(gold_a, dim4(3, 2), d);
}