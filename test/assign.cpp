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

#include <half.hpp>

#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype_traits;
using af::end;
using af::exception;
using af::randu;
using af::seq;
using af::span;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class ArrayAssign : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat1D.push_back(af_make_seq(5, 20, 1));

        subMat2D.push_back(af_make_seq(1, 2, 1));
        subMat2D.push_back(af_make_seq(1, 2, 1));

        subMat3D.push_back(af_make_seq(3, 4, 1));
        subMat3D.push_back(af_make_seq(0, 1, 1));
        subMat3D.push_back(af_make_seq(1, 2, 1));

        subMat4D.push_back(af_make_seq(3, 4, 1));
        subMat4D.push_back(af_make_seq(0, 1, 1));
        subMat4D.push_back(af_make_seq(0, 1, 1));
        subMat4D.push_back(af_make_seq(1, 2, 1));

        subMat1D_to_2D.push_back(af_make_seq(1, 2, 1));
        subMat1D_to_2D.push_back(af_make_seq(1, 1, 1));

        subMat1D_to_3D.push_back(af_make_seq(5, 20, 1));
        subMat1D_to_3D.push_back(af_make_seq(1, 1, 1));
        subMat1D_to_3D.push_back(af_make_seq(2, 2, 1));

        subMat2D_to_3D.push_back(af_make_seq(3, 4, 1));
        subMat2D_to_3D.push_back(af_make_seq(0, 1, 1));
        subMat2D_to_3D.push_back(af_make_seq(1, 1, 1));

        subMat1D_to_4D.push_back(af_make_seq(3, 4, 1));
        subMat1D_to_4D.push_back(af_make_seq(0, 0, 1));
        subMat1D_to_4D.push_back(af_make_seq(0, 0, 1));
        subMat1D_to_4D.push_back(af_make_seq(1, 1, 1));

        subMat2D_to_4D.push_back(af_make_seq(3, 4, 1));
        subMat2D_to_4D.push_back(af_make_seq(0, 1, 1));
        subMat2D_to_4D.push_back(af_make_seq(0, 0, 1));
        subMat2D_to_4D.push_back(af_make_seq(1, 1, 1));

        subMat3D_to_4D.push_back(af_make_seq(3, 4, 1));
        subMat3D_to_4D.push_back(af_make_seq(0, 1, 1));
        subMat3D_to_4D.push_back(af_make_seq(0, 1, 1));
        subMat3D_to_4D.push_back(af_make_seq(1, 1, 1));
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
typedef ::testing::Types<float, cdouble, cfloat, double, int, uint, char, uchar,
                         intl, uintl, short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(ArrayAssign, TestTypes);

template<typename inType, typename outType>
void assignTest(string pTestFile, const vector<af_seq> *seqv) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTests<inType, outType, int>(pTestFile, numDims, in, tests);

    dim4 dims0        = numDims[0];
    dim4 dims1        = numDims[1];
    af_array lhsArray = 0;
    af_array rhsArray = 0;
    af_array outArray = 0;

    ASSERT_SUCCESS(af_create_array(&rhsArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<inType>::af_type));

    ASSERT_SUCCESS(af_create_array(&lhsArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<outType>::af_type));

    ASSERT_SUCCESS(af_assign_seq(&outArray, lhsArray, seqv->size(),
                                 &seqv->front(), rhsArray));

    outType *outData = new outType[dims1.elements()];

    ASSERT_SUCCESS(af_get_data_ptr((void *)outData, outArray));

    vector<outType> currGoldBar = tests[0];
    size_t nElems               = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outData;
    ASSERT_SUCCESS(af_release_array(rhsArray));
    ASSERT_SUCCESS(af_release_array(lhsArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

template<typename T>
void assignTestCPP(string pTestFile, const vector<af_seq> &seqv) {
    SUPPORTED_TYPE_CHECK(T);
    try {
        vector<dim4> numDims;
        vector<vector<T>> in;
        vector<vector<T>> tests;

        readTests<T, T, int>(pTestFile, numDims, in, tests);

        dim4 dims0 = numDims[0];
        dim4 dims1 = numDims[1];

        array a(dims0, &(in[0].front()));
        array b(dims1, &(in[1].front()));

        switch (seqv.size()) {
            case 1: b(seqv[0]) = a; break;
            case 2: b(seqv[0], seqv[1]) = a; break;
            case 3: b(seqv[0], seqv[1], seqv[2]) = a; break;
            case 4: b(seqv[0], seqv[1], seqv[2], seqv[3]) = a; break;
            default: assert(1 != 1 && "Does not compute");
        }

        T *outData = new T[dims1.elements()];
        b.host(outData);

        vector<T> currGoldBar = tests[0];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            EXPECT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
        delete[] outData;
    } catch (const exception &ex) {
        FAIL() << "Exception thrown: " << ex.what();
    }
}

TYPED_TEST(ArrayAssign, Vector) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/1d_to_1d.test"),
                                     &(this->subMat1D));
}

TYPED_TEST(ArrayAssign, VectorCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/1d_to_1d.test"),
                             this->subMat1D);
}

TYPED_TEST(ArrayAssign, Matrix) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/2d_to_2d.test"),
                                     &(this->subMat2D));
}

TYPED_TEST(ArrayAssign, MatrixCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/2d_to_2d.test"),
                             this->subMat2D);
}

TYPED_TEST(ArrayAssign, Cube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/3d_to_3d.test"),
                                     &(this->subMat3D));
}

TYPED_TEST(ArrayAssign, CubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/3d_to_3d.test"),
                             this->subMat3D);
}

TYPED_TEST(ArrayAssign, HyperCube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/4d_to_4d.test"),
                                     &(this->subMat4D));
}

TYPED_TEST(ArrayAssign, HyperCubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/4d_to_4d.test"),
                             this->subMat4D);
}

TYPED_TEST(ArrayAssign, Vector2Matrix) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/1d_to_2d.test"),
                                     &(this->subMat1D_to_2D));
}

TYPED_TEST(ArrayAssign, Vector2MatrixCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/1d_to_2d.test"),
                             this->subMat1D_to_2D);
}

TYPED_TEST(ArrayAssign, Vector2Cube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/1d_to_3d.test"),
                                     &(this->subMat1D_to_3D));
}

TYPED_TEST(ArrayAssign, Vector2CubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/1d_to_3d.test"),
                             this->subMat1D_to_3D);
}

TYPED_TEST(ArrayAssign, Matrix2Cube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/2d_to_3d.test"),
                                     &(this->subMat2D_to_3D));
}

TYPED_TEST(ArrayAssign, Matrix2CubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/2d_to_3d.test"),
                             this->subMat2D_to_3D);
}

TYPED_TEST(ArrayAssign, Vector2HyperCube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/1d_to_4d.test"),
                                     &(this->subMat1D_to_4D));
}

TYPED_TEST(ArrayAssign, Vector2HyperCubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/1d_to_4d.test"),
                             this->subMat1D_to_4D);
}

TYPED_TEST(ArrayAssign, Matrix2HyperCube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/2d_to_4d.test"),
                                     &(this->subMat2D_to_4D));
}

TYPED_TEST(ArrayAssign, Matrix2HyperCubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/2d_to_4d.test"),
                             this->subMat2D_to_4D);
}

TYPED_TEST(ArrayAssign, Cube2HyperCube) {
    assignTest<TypeParam, TypeParam>(string(TEST_DIR "/assign/3d_to_4d.test"),
                                     &(this->subMat3D_to_4D));
}

TYPED_TEST(ArrayAssign, Cube2HyperCubeCPP) {
    assignTestCPP<TypeParam>(string(TEST_DIR "/assign/3d_to_4d.test"),
                             this->subMat3D_to_4D);
}

template<typename T>
void assignScalarCPP(string pTestFile, const vector<af_seq> &seqv) {
    SUPPORTED_TYPE_CHECK(T);
    try {
        vector<dim4> numDims;
        vector<vector<T>> in;
        vector<vector<T>> tests;

        readTests<T, T, int>(pTestFile, numDims, in, tests);

        dim4 dims1 = numDims[1];

        T a = in[0][0];
        array b(dims1, &(in[1].front()));

        switch (seqv.size()) {
            case 1: b(seqv[0]) = a; break;
            case 2: b(seqv[0], seqv[1]) = a; break;
            case 3: b(seqv[0], seqv[1], seqv[2]) = a; break;
            case 4: b(seqv[0], seqv[1], seqv[2], seqv[3]) = a; break;
            default: assert(1 != 1 && "Does not compute");
        }

        T *outData = new T[dims1.elements()];
        b.host(outData);

        vector<T> currGoldBar = tests[0];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            if (currGoldBar[elIter] != outData[elIter]) {
                switch (seqv.size()) {
                    case 1: printf("b(seqv[0]) = a\n"); break;
                    case 2: printf("b(seqv[0],seqv[1]) = a\n"); break;
                    case 3: printf("b(seqv[0],seqv[1], seqv[2]) = a\n"); break;
                    case 4:
                        printf("b(seqv[0],seqv[1], seqv[2], seqv[3]) = a\n");
                        break;
                    default: assert(1 != 1 && "Does not compute");
                }
                cout << "a: " << a << endl;
                af_print(b);
                ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                    << "at: " << elIter << endl;
            }
        }
        delete[] outData;
    } catch (const exception &ex) {
        FAIL() << "Exception thrown: " << ex.what();
    }
}

TYPED_TEST(ArrayAssign, Scalar1DCPP) {
    assignScalarCPP<TypeParam>(string(TEST_DIR "/assign/scalar_to_1d.test"),
                               this->subMat1D);
}

TYPED_TEST(ArrayAssign, Scalar2DCPP) {
    assignScalarCPP<TypeParam>(string(TEST_DIR "/assign/scalar_to_2d.test"),
                               this->subMat2D);
}

TYPED_TEST(ArrayAssign, Scalar3DCPP) {
    assignScalarCPP<TypeParam>(string(TEST_DIR "/assign/scalar_to_3d.test"),
                               this->subMat3D);
}

TYPED_TEST(ArrayAssign, Scalar4DCPP) {
    assignScalarCPP<TypeParam>(string(TEST_DIR "/assign/scalar_to_4d.test"),
                               this->subMat4D);
}

TYPED_TEST(ArrayAssign, AssignRowCPP) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const int dimsize = 10;
    vector<TypeParam> input(100, TypeParam(1.0));
    vector<TypeParam> sq(dimsize);
    vector<int> arIdx(2);
    for (int i = 0; i < (int)sq.size(); i++) sq[i] = i;
    arIdx[0] = 5;
    arIdx[1] = 7;

    array in(dimsize, dimsize, &input.front(), afHost);
    dim4 size(dimsize, 1, 1, 1);
    array sarr(size, &sq.front(), afHost);
    array arrIdx(2, &arIdx.front(), afHost);

    in.row(0)        = sarr;
    in.row(2)        = 2;
    in(arrIdx, span) = 8;
    in.row(end)      = 3;
    in.rows(3, 4)    = 7;

    vector<TypeParam> out(100);
    in.host(&out.front());

    for (int col = 0; col < dimsize; col++) {
        for (int row = 0; row < dimsize; row++) {
            if (row == 0)
                ASSERT_EQ(sq[col], out[col * dimsize + row])
                    << "Assigning array to indexed array using col";
            else if (row == 2)
                ASSERT_EQ(TypeParam(2), out[col * dimsize + row])
                    << "Assigning value to indexed array using col";
            else if (row == dimsize - 1)
                ASSERT_EQ(TypeParam(3), out[col * dimsize + row])
                    << "Assigning value to array which is indexed using end.";
            else if (row == 3 || row == 4)
                ASSERT_EQ(TypeParam(7), out[col * dimsize + row])
                    << "Assigning value to an array which is indexed using an "
                       "rows";
            else if (row == 5 || row == 7)
                ASSERT_EQ(TypeParam(8), out[col * dimsize + row])
                    << "Assigning value to an array which is indexed using an "
                       "array (i.e. in(arrIdx, span) = 8);) using row";
            else
                ASSERT_EQ(TypeParam(1), out[col * dimsize + row])
                    << "Values written to incorrect location";
        }
    }
}

TYPED_TEST(ArrayAssign, AssignColumnCPP) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const int dimsize = 10;
    vector<TypeParam> input(100, TypeParam(1.0));
    vector<TypeParam> sq(dimsize);
    vector<int> arIdx(2);
    for (int i = 0; i < (int)sq.size(); i++) sq[i] = i;
    arIdx[0] = 5;
    arIdx[1] = 7;

    array in(dimsize, dimsize, &input.front(), afHost);
    dim4 size(dimsize, 1, 1, 1);
    array sarr(size, &sq.front(), afHost);
    array arrIdx(2, &arIdx.front(), afHost);

    in.col(0)        = sarr;
    in.col(2)        = 2;
    in(span, arrIdx) = 8;
    in.col(end)      = 3;
    in.cols(3, 4)    = 7;

    vector<TypeParam> out(100);
    in.host(&out.front());

    for (int col = 0; col < dimsize; col++) {
        for (int row = 0; row < dimsize; row++) {
            if (col == 0)
                ASSERT_EQ(sq[row], out[col * dimsize + row])
                    << "Assigning array to indexed array using col";
            else if (col == 2)
                ASSERT_EQ(TypeParam(2), out[col * dimsize + row])
                    << "Assigning value to indexed array using col";
            else if (col == dimsize - 1)
                ASSERT_EQ(TypeParam(3), out[col * dimsize + row])
                    << "Assigning value to array which is indexed using end.";
            else if (col == 3 || col == 4)
                ASSERT_EQ(TypeParam(7), out[col * dimsize + row])
                    << "Assigning value to an array which is indexed using an "
                       "cols";
            else if (col == 5 || col == 7)
                ASSERT_EQ(TypeParam(8), out[col * dimsize + row])
                    << "Assigning value to an array which is indexed using an "
                       "array (i.e. in(span, arrIdx) = 8);) using col";
            else
                ASSERT_EQ(TypeParam(1), out[col * dimsize + row])
                    << "Values written to incorrect location";
        }
    }
}

TYPED_TEST(ArrayAssign, AssignSliceCPP) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    const int dimsize = 10;
    vector<TypeParam> input(1000, TypeParam(1.0));
    vector<TypeParam> sq(dimsize * dimsize);
    vector<int> arIdx(2);
    for (int i = 0; i < (int)sq.size(); i++) sq[i] = i;
    arIdx[0] = 5;
    arIdx[1] = 7;

    array in(dimsize, dimsize, dimsize, &input.front(), afHost);
    dim4 size(dimsize, dimsize, 1, 1);
    array sarr(size, &sq.front(), afHost);
    array arrIdx(2, &arIdx.front(), afHost);

    in.slice(0)            = sarr;
    in.slice(2)            = 2;
    in(span, span, arrIdx) = 8;
    in.slice(end)          = 3;
    in.slices(3, 4)        = 7;

    vector<TypeParam> out(1000);
    in.host(&out.front());

    for (int slice = 0; slice < dimsize; slice++) {
        for (int col = 0; col < dimsize; col++) {
            for (int row = 0; row < dimsize; row++) {
                int idx = slice * dimsize * dimsize + col * dimsize + row;
                if (slice == 0)
                    ASSERT_EQ(sq[col * dimsize + row], out[idx])
                        << "Assigning array to indexed array using col";
                else if (slice == 2)
                    ASSERT_EQ(TypeParam(2), out[idx])
                        << "Assigning value to indexed array using col";
                else if (slice == dimsize - 1)
                    ASSERT_EQ(TypeParam(3), out[idx])
                        << "Assigning value to array which is indexed using "
                           "end.";
                else if (slice == 3 || slice == 4)
                    ASSERT_EQ(TypeParam(7), out[idx])
                        << "Assigning value to an array which is indexed using "
                           "an slices";
                else if (slice == 5 || slice == 7)
                    ASSERT_EQ(TypeParam(8), out[idx])
                        << "Assigning value to an array which is indexed using "
                           "an array (i.e. in(span, span, arrIdx) = 8);) using "
                           "slice";
                else
                    ASSERT_EQ(TypeParam(1), out[idx])
                        << "Values written to incorrect location";
            }
        }
    }
}

TEST(ArrayAssign, InvalidArgs) {
    vector<cfloat> in(100, cfloat(0, 0));
    vector<float> tests(100, float(1));

    dim4 dims0(10, 1, 1, 1);
    dim4 dims1(100, 1, 1, 1);
    af_array lhsArray = 0;
    af_array rhsArray = 0;
    af_array outArray = 0;

    vector<af_seq> seqv;
    seqv.push_back(af_make_seq(5, 14, 1));

    ASSERT_EQ(AF_ERR_ARG, af_assign_seq(&outArray, lhsArray, seqv.size(),
                                        &seqv.front(), rhsArray));

    ASSERT_SUCCESS(af_create_array(&rhsArray, &(in.front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<cfloat>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_assign_seq(&outArray, lhsArray, seqv.size(),
                                        &seqv.front(), rhsArray));

    ASSERT_SUCCESS(af_create_array(&lhsArray, &(in.front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG,
              af_assign_seq(&outArray, lhsArray, 0, &seqv.front(), rhsArray));

    ASSERT_EQ(AF_ERR_TYPE, af_assign_seq(&outArray, lhsArray, seqv.size(),
                                         &seqv.front(), rhsArray));

    ASSERT_SUCCESS(af_release_array(rhsArray));
    ASSERT_SUCCESS(af_release_array(lhsArray));
}

TEST(ArrayAssign, CPP_ASSIGN_TO_INDEXED) {
    vector<int> in(20);
    for (int i = 0; i < (int)in.size(); i++) in[i] = i;

    array input(10, 2, &in.front(), afHost);

    // Tests array_proxy to array_proxy assignment
    input(span, 0) = input(span, 1);

    vector<int> out(20);
    input.host(&out.front());

    for (int i = 0; i < 10; i++) ASSERT_EQ(i + 10, out[i]);
    for (int i = 10; i < (int)in.size(); i++) ASSERT_EQ(i, out[i]);
}

TEST(ArrayAssign, CPP_END) {
    const int n       = 5;
    const int m       = 5;
    const int end_off = 2;

    array a                = randu(n, m);
    array b                = randu(1, m);
    a(end - end_off, span) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < m; i++) { ASSERT_EQ(hA[i * n + end_off], hB[i]); }

    af_free_host(hA);
    af_free_host(hB);
}

TEST(ArrayAssign, CPP_END_SEQ) {
    const int num       = 20;
    const int end_begin = 10;
    const int end_end   = 0;
    const int len       = end_begin - end_end + 1;

    array a                                = randu(num);
    array b                                = randu(len);
    a(seq(end - end_begin, end - end_end)) = b;

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < len; i++) { ASSERT_EQ(hA[i + end_begin - 1], hB[i]); }

    af_free_host(hA);
    af_free_host(hB);
}

TEST(ArrayAssign, CPP_COPY_ON_WRITE) {
    const int num = 20;
    const int len = 10;

    array a    = randu(num);
    float *hAO = a.host<float>();

    array a_copy = a;
    array b      = randu(len);
    a(seq(len))  = b;

    float *hA  = a.host<float>();
    float *hB  = b.host<float>();
    float *hAC = a_copy.host<float>();

    // first half should be from B
    for (int i = 0; i < len; i++) { ASSERT_EQ(hA[i], hB[i]); }

    // Second half should be same as original
    for (int i = 0; i < num - len; i++) {
        ASSERT_EQ(hA[i + len], hAO[i + len]);
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) { ASSERT_EQ(hAO[i], hAC[i]); }

    af_free_host(hA);
    af_free_host(hB);
    af_free_host(hAC);
    af_free_host(hAO);
}

TEST(ArrayAssign, CPP_ASSIGN_BINOP) {
    const int num = 20;
    const int len = 10;

    array a    = randu(num);
    float *hAO = a.host<float>();

    array a_copy = a;
    array b      = randu(len);
    a(seq(len)) += b;

    float *hA  = a.host<float>();
    float *hB  = b.host<float>();
    float *hAC = a_copy.host<float>();

    // first half should be hAO + hB
    for (int i = 0; i < len; i++) { ASSERT_EQ(hA[i], hAO[i] + hB[i]); }

    // Second half should be same as original
    for (int i = 0; i < num - len; i++) {
        ASSERT_EQ(hA[i + len], hAO[i + len]);
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) { ASSERT_EQ(hAO[i], hAC[i]); }

    af_free_host(hA);
    af_free_host(hB);
    af_free_host(hAC);
    af_free_host(hAO);
}

TEST(ArrayAssign, CPP_ASSIGN_VECTOR) {
    const int num = 20;

    array a = randu(1, num);
    array b = randu(num);

    array c, idx;
    sort(c, idx, b);

    a(idx) = c;

    ASSERT_EQ(a.dims(0), (dim_t)1);
    ASSERT_EQ(a.dims(1), (dim_t)num);
    ASSERT_EQ(c.dims(0), (dim_t)num);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();

    for (int i = 0; i < num; i++) { ASSERT_EQ(h_a[i], h_b[i]) << "at " << i; }

    af_free_host(h_a);
    af_free_host(h_b);
}

TEST(ArrayAssign, CPP_ASSIGN_VECTOR_SEQ) {
    const int num = 20;
    const int len = 10;
    const int st  = 3;
    const int en  = st + len - 1;

    array a  = randu(1, 1, num);
    array a0 = a;
    array b  = randu(len);

    array idx = seq(st, en);

    a(seq(st, en)) = b;

    ASSERT_EQ(a.dims(0), (dim_t)1);
    ASSERT_EQ(a.dims(1), (dim_t)1);
    ASSERT_EQ(a.dims(2), (dim_t)num);
    ASSERT_EQ(b.dims(0), (dim_t)len);

    float *h_a0 = a0.host<float>();
    float *h_a  = a.host<float>();
    float *h_b  = b.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) {
            ASSERT_EQ(h_a[i], h_b[i - st]);
        } else {
            ASSERT_EQ(h_a[i], h_a0[i]);
        }
    }

    af_free_host(h_a0);
    af_free_host(h_a);
    af_free_host(h_b);
}

TEST(ArrayAssign, CPP_ASSIGN_VECTOR_2D) {
    const int nx  = 4;
    const int ny  = 5;
    const int num = nx * ny;

    array a = randu(nx, ny);
    array b = randu(num);

    array c, idx;
    sort(c, idx, b);

    a(idx) = c;

    ASSERT_EQ(a.dims(0), (dim_t)nx);
    ASSERT_EQ(a.dims(1), (dim_t)ny);
    ASSERT_EQ(c.dims(0), (dim_t)num);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();

    for (int i = 0; i < num; i++) { ASSERT_EQ(h_a[i], h_b[i]) << "at " << i; }

    af_free_host(h_a);
    af_free_host(h_b);
}

TEST(ArrayAssign, CPP_ASSIGN_VECTOR_SEQ_2D) {
    const int nx  = 4;
    const int nz  = 5;
    const int num = nx * nz;
    const int len = 10;
    const int st  = 3;
    const int en  = st + len - 1;

    array a  = randu(nx, 1, nz);
    array a0 = a;
    array b  = randu(len);

    a(seq(st, en)) = b;

    ASSERT_EQ(a.dims(0), (dim_t)nx);
    ASSERT_EQ(a.dims(1), (dim_t)1);
    ASSERT_EQ(a.dims(2), (dim_t)nz);
    ASSERT_EQ(b.dims(0), (dim_t)len);

    float *h_a0 = a0.host<float>();
    float *h_a  = a.host<float>();
    float *h_b  = b.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) {
            ASSERT_EQ(h_a[i], h_b[i - st]);
        } else {
            ASSERT_EQ(h_a[i], h_a0[i]);
        }
    }

    af_free_host(h_a0);
    af_free_host(h_a);
    af_free_host(h_b);
}

TEST(Assign, Copy) {
    const int num = 20;
    const int len = 10;
    const int st  = 3;
    const int en  = st + len - 1;

    array a     = randu(num, 1);
    float *h_a0 = a.host<float>();

    array b = randu(len);

    float *d_ptr = a.device<float>();
    copy(a, b, seq(st, en));

    // Ensure that a still has same device pointer
    ASSERT_EQ(d_ptr, a.device<float>());

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) {
            ASSERT_EQ(h_a[i], h_b[i - st]);
        } else {
            ASSERT_EQ(h_a[i], h_a0[i]);
        }
    }

    af_free_host(h_a0);
    af_free_host(h_a);
    af_free_host(h_b);
}

TEST(Asssign, LinearCPP) {
    const int nx    = 5;
    const int ny    = 4;
    const float val = 3;

    const int st = nx - 2;
    const int en = nx * (ny - 1);

    array a       = randu(nx, ny);
    array a_copy  = a;
    af::index idx = seq(st, en);
    a(idx)        = 3;

    ASSERT_EQ(a.dims(0), a_copy.dims(0));
    ASSERT_EQ(a.dims(1), a_copy.dims(1));

    vector<float> ha(nx * ny);
    vector<float> ha_copy(nx * ny);

    a.host(&ha[0]);
    a_copy.host(&ha_copy[0]);

    for (int i = 0; i < nx * ny; i++) {
        if (i < st || i > en)
            ASSERT_EQ(ha[i], ha_copy[i]) << "at " << i;
        else
            ASSERT_EQ(ha[i], val) << "at " << i;
    }
}

TEST(Asssign, LinearCPPMaxDim) {
    const size_t largeDim = 65535 * 32 + 2;
    const float val       = 3;

    array a       = randu(1, 2 * largeDim);
    array a_copy  = a.copy();
    af::index idx = array(seq(10, largeDim + 10));
    a(span, idx)  = val;

    ASSERT_EQ(a.dims(0), a_copy.dims(0));

    vector<float> ha(2 * largeDim);
    vector<float> ha_copy(2 * largeDim);

    a.host(&ha[0]);
    a_copy.host(&ha_copy[0]);

    for (unsigned int i = 0; i < 2 * largeDim; i++) {
        if (i >= 10 && i <= largeDim + 10) {
            ASSERT_EQ(ha[i], val) << "at " << i;
        } else {
            ASSERT_EQ(ha[i], ha_copy[i]) << "at " << i;
        }
    }
}

TEST(Asssign, LinearAssignSeq) {
    const int nx    = 5;
    const int ny    = 4;
    const float val = 3;
    const array rhs = constant(val, 1, 1);

    const int st = nx - 2;
    const int en = nx * (ny - 1);

    array a       = randu(nx, ny);
    af::index idx = seq(st, en);

    af_array in_arr  = a.get();
    af_index_t ii    = idx.get();
    af_array rhs_arr = rhs.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_assign_seq(&out_arr, in_arr, 1, &ii.idx.seq, rhs_arr));

    array out(out_arr);

    ASSERT_EQ(a.dims(0), out.dims(0));
    ASSERT_EQ(a.dims(1), out.dims(1));

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < nx * ny; i++) {
        if (i < st || i > en)
            ASSERT_EQ(hout[i], ha[i]) << "at " << i;
        else
            ASSERT_EQ(hout[i], val) << "at " << i;
    }
}

TEST(Asssign, LinearAssignGenSeq) {
    const int nx    = 5;
    const int ny    = 4;
    const float val = 3;
    const array rhs = constant(val, 1, 1);

    const int st = nx - 2;
    const int en = nx * (ny - 1);

    array a       = randu(nx, ny);
    af::index idx = seq(st, en);

    af_array in_arr  = a.get();
    af_index_t ii    = idx.get();
    af_array rhs_arr = rhs.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_assign_gen(&out_arr, in_arr, 1, &ii, rhs_arr));

    array out(out_arr);

    ASSERT_EQ(a.dims(0), out.dims(0));
    ASSERT_EQ(a.dims(1), out.dims(1));

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < nx * ny; i++) {
        if (i < st || i > en)
            ASSERT_EQ(hout[i], ha[i]) << "at " << i;
        else
            ASSERT_EQ(hout[i], val) << "at " << i;
    }
}

TEST(Asssign, LinearAssignGenArr) {
    const int nx    = 5;
    const int ny    = 4;
    const float val = 3;
    const array rhs = constant(val, 1, 1);

    const int st = nx - 2;
    const int en = nx * (ny - 1);

    array a       = randu(nx, ny);
    af::index idx = array(seq(st, en));

    af_array in_arr  = a.get();
    af_index_t ii    = idx.get();
    af_array rhs_arr = rhs.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_assign_gen(&out_arr, in_arr, 1, &ii, rhs_arr));

    array out(out_arr);

    ASSERT_EQ(a.dims(0), out.dims(0));
    ASSERT_EQ(a.dims(1), out.dims(1));

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < nx * ny; i++) {
        if (i < st || i > en)
            ASSERT_EQ(hout[i], ha[i]) << "at " << i;
        else
            ASSERT_EQ(hout[i], val) << "at " << i;
    }
}

TEST(Assign, ISSUE_1764) {
    int x   = 2;
    int y   = 2;
    int z   = 2;
    array a = randu(x, y, z);
    vector<float> ha0(a.elements());
    a.host(&ha0[0]);
    a(0, span, span) = a(1, span, span);
    vector<float> ha1(a.elements());
    a.host(&ha1[0]);
    for (int k = 0; k < z; k++) {
        for (int j = 0; j < y; j++) {
            int offset = (j + k * y) * x;
            ASSERT_EQ(ha0[offset + 1], ha1[offset + 0]);
            ASSERT_EQ(ha0[offset + 1], ha1[offset + 1]);
        }
    }
}

TEST(Assign, ISSUE_1677) {
    try {
        dim_t sz      = 1;
        array a       = constant(1.0f, 3, sz, f32);
        array b       = constant(2.0f, 3, sz, f32);
        array cond    = constant(0, sz, b8);  // all false
        a(span, cond) = b(span, cond);
    } catch (exception &ex) {
        FAIL() << "ArrayFire exception: " << ex.what();
    } catch (...) { FAIL() << "Unknown exception thrown"; }
}

TEST(Index, ISSUE_2533) {
    int elements = 5 * 10;
    std::vector<float> gold(elements, 0);

    int assigned_elements = 5 * 6;
    for (int i = 0; i < assigned_elements; i++) { gold[i] = 1; }

    af::array a = constant(0, 5, 10);
    af::array b = constant(1, 5, 10);

    a(af::span, af::seq(0, 5)) = b(af::span, af::seq(0, 5));

    ASSERT_VEC_ARRAY_EQ(gold, dim4(5, 10), a);
}
