/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class ArrayAssign : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat1D.push_back(af_make_seq(5,20,1));

            subMat2D.push_back(af_make_seq(1,2,1));
            subMat2D.push_back(af_make_seq(1,2,1));

            subMat3D.push_back(af_make_seq(3,4,1));
            subMat3D.push_back(af_make_seq(0,1,1));
            subMat3D.push_back(af_make_seq(1,2,1));

            subMat4D.push_back(af_make_seq(3,4,1));
            subMat4D.push_back(af_make_seq(0,1,1));
            subMat4D.push_back(af_make_seq(0,1,1));
            subMat4D.push_back(af_make_seq(1,2,1));

            subMat1D_to_2D.push_back(af_make_seq(1,2,1));
            subMat1D_to_2D.push_back(af_make_seq(1,1,1));

            subMat1D_to_3D.push_back(af_make_seq(5,20,1));
            subMat1D_to_3D.push_back(af_make_seq(1,1,1));
            subMat1D_to_3D.push_back(af_make_seq(2,2,1));

            subMat2D_to_3D.push_back(af_make_seq(3,4,1));
            subMat2D_to_3D.push_back(af_make_seq(0,1,1));
            subMat2D_to_3D.push_back(af_make_seq(1,1,1));

            subMat1D_to_4D.push_back(af_make_seq(3,4,1));
            subMat1D_to_4D.push_back(af_make_seq(0,0,1));
            subMat1D_to_4D.push_back(af_make_seq(0,0,1));
            subMat1D_to_4D.push_back(af_make_seq(1,1,1));

            subMat2D_to_4D.push_back(af_make_seq(3,4,1));
            subMat2D_to_4D.push_back(af_make_seq(0,1,1));
            subMat2D_to_4D.push_back(af_make_seq(0,0,1));
            subMat2D_to_4D.push_back(af_make_seq(1,1,1));

            subMat3D_to_4D.push_back(af_make_seq(3,4,1));
            subMat3D_to_4D.push_back(af_make_seq(0,1,1));
            subMat3D_to_4D.push_back(af_make_seq(0,1,1));
            subMat3D_to_4D.push_back(af_make_seq(1,1,1));
        }
        vector<af_seq> subMat1D;

        vector<af_seq> subMat2D;
        vector<af_seq> subMat1D_to_2D;

        vector<af_seq> subMat3D;
        vector<af_seq> subMat1D_to_3D;
        vector<af_seq> subMat2D_to_3D;

        vector<af_seq> subMat4D;
        vector<af_seq> subMat1D_to_4D;
        vector<af_seq> subMat2D_to_4D;
        vector<af_seq> subMat3D_to_4D;
};

// create a list of types to be tested
typedef ::testing::Types<af::cdouble, af::cfloat, double, float, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(ArrayAssign, TestTypes);

template<typename inType, typename outType>
void assignTest(string pTestFile, const vector<af_seq> *seqv)
{
    if (noDoubleTests<inType>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4>  numDims;
    vector<vector<inType> >      in;
    vector<vector<outType> >   tests;

    readTests<inType, outType, int>(pTestFile, numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af_array lhsArray  = 0;
    af_array rhsArray  = 0;
    af_array outArray  = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&rhsArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&lhsArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<outType>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_assign(&outArray, lhsArray, seqv->size(), &seqv->front(), rhsArray));

    outType *outData = new outType[dims1.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<outType> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(rhsArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(lhsArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TYPED_TEST(ArrayAssign, Vector)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/1d_to_1d.test"), &(this->subMat1D));
}

TYPED_TEST(ArrayAssign, Matrix)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/2d_to_2d.test"), &(this->subMat2D));
}

TYPED_TEST(ArrayAssign, Cube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/3d_to_3d.test"), &(this->subMat3D));
}

TYPED_TEST(ArrayAssign, HyperCube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/4d_to_4d.test"), &(this->subMat4D));
}

TYPED_TEST(ArrayAssign, Vector2Matrix)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/1d_to_2d.test"), &(this->subMat1D_to_2D));
}

TYPED_TEST(ArrayAssign, Vector2Cube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/1d_to_3d.test"), &(this->subMat1D_to_3D));
}

TYPED_TEST(ArrayAssign, Matrix2Cube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/2d_to_3d.test"), &(this->subMat2D_to_3D));
}

TYPED_TEST(ArrayAssign, Vector2HyperCube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/1d_to_4d.test"), &(this->subMat1D_to_4D));
}

TYPED_TEST(ArrayAssign, Matrix2HyperCube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/2d_to_4d.test"), &(this->subMat2D_to_4D));
}

TYPED_TEST(ArrayAssign, Cube2HyperCube)
{
    assignTest<TypeParam, TypeParam>(string(TEST_DIR"/assign/3d_to_4d.test"), &(this->subMat3D_to_4D));
}

TEST(ArrayAssign, InvalidArgs)
{
    vector<af::cfloat> in(10, af::cfloat(0,0));
    vector<float> tests(100, float(1));

    af::dim4 dims0(10, 1, 1, 1);
    af::dim4 dims1(100, 1, 1, 1);
    af_array lhsArray = 0;
    af_array rhsArray = 0;
    af_array outArray = 0;

    vector<af_seq> seqv;
    seqv.push_back(af_make_seq(5,14,1));

    ASSERT_EQ(AF_ERR_ARG, af_assign(&outArray,
                                    lhsArray, seqv.size(), &seqv.front(), rhsArray));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&rhsArray, &(in.front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<af::cfloat>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_assign(&outArray,
                                    lhsArray, seqv.size(), &seqv.front(), rhsArray));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&lhsArray, &(in.front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_assign(&outArray, lhsArray, 0, &seqv.front(), rhsArray));

    ASSERT_EQ(AF_ERR_INVALID_TYPE, af_assign(&outArray,
                                             lhsArray, seqv.size(), &seqv.front(), rhsArray));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(rhsArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(lhsArray));
}

TEST(ArrayAssign, CPP)
{
    if (noDoubleTests<float>()) return;

    using af::array;

    vector<af_seq> seqv;
    seqv.push_back(af_make_seq(1,2,1));
    seqv.push_back(af_make_seq(1,2,1));

    vector<af::dim4>  numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/assign/2d_to_2d.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];

    array a(dims0, &(in[0].front()));
    array b(dims1, &(in[1].front()));

    b(seqv[0],seqv[1]) = a;

    float *outData = new float[dims1.elements()];
    b.host(outData);

    vector<float> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(ArrayAssign, CPP_END)
{
    using af::array;

    const int n = 5;
    const int m = 5;
    const int end_off = 2;

    array a = af::randu(n, m);
    array b = af::randu(1, m);
    a(af::end - end_off, af::span) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < m; i++) {
        ASSERT_EQ(hA[i * n + end_off], hB[i]);
    }


    delete[] hA;
    delete[] hB;
}

TEST(ArrayAssign, CPP_END_SEQ)
{
    using af::array;

    const int num = 20;
    const int end_begin = 10;
    const int end_end = 0;
    const int len = end_begin - end_end + 1;

    array a = af::randu(num);
    array b = af::randu(len);
    a(af::seq(af::end - end_begin, af::end - end_end)) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < len; i++) {
        ASSERT_EQ(hA[i + end_begin - 1], hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

TEST(ArrayAssign, CPP_COPY_ON_WRITE)
{
    using af::array;

    const int num = 20;
    const int len = 10;

    array a = af::randu(num);
    float *hAO = a.host<float>();

    array a_copy = a;
    array b = af::randu(len);
    a(af::seq(len)) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();
    float *hAC = a_copy.host<float>();

    // first half should be from B
    for (int i = 0; i < len; i++) {
        ASSERT_EQ(hA[i], hB[i]);
    }

    // Second half should be same as original
    for (int i = 0; i < num - len; i++) {
        ASSERT_EQ(hA[i + len], hAO[i + len]);
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hAO[i], hAC[i]);
    }

    delete[] hA;
    delete[] hB;
    delete[] hAC;
    delete[] hAO;
}

TEST(ArrayAssign, CPP_ASSIGN_BINOP)
{
    using af::array;

    const int num = 20;
    const int len = 10;

    array a = af::randu(num);
    float *hAO = a.host<float>();

    array a_copy = a;
    array b = af::randu(len);
    a(af::seq(len)) += b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();
    float *hAC = a_copy.host<float>();

    // first half should be hAO + hB
    for (int i = 0; i < len; i++) {
        ASSERT_EQ(hA[i], hAO[i] + hB[i]);
    }

    // Second half should be same as original
    for (int i = 0; i < num - len; i++) {
        ASSERT_EQ(hA[i + len], hAO[i + len]);
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hAO[i], hAC[i]);
    }

    delete[] hA;
    delete[] hB;
    delete[] hAC;
    delete[] hAO;
}
