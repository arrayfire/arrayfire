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
#include <iostream>
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
                         short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_CASE(Moddims, TestTypes);

template<typename T>
void moddimsTest(string pTestFile, bool isSubRef = false,
                 const vector<af_seq> *seqv = NULL) {
    if (noDoubleTests<T>()) return;

    vector<dim4> numDims;

    vector<vector<T> > in;
    vector<vector<T> > tests;
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
    if (noDoubleTests<T>()) return;

    vector<dim4> numDims;

    vector<vector<T> > in;
    vector<vector<T> > tests;
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
    if (noDoubleTests<T>()) return;

    vector<dim4> numDims;

    vector<vector<T> > in;
    vector<vector<T> > tests;
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
    if (noDoubleTests<T>()) return;

    vector<dim4> numDims;

    vector<vector<T> > in;
    vector<vector<T> > tests;
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
