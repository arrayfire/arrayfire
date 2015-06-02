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
using af::cfloat;
using af::cdouble;

template<typename T>
class Transpose : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat2D.push_back(af_make_seq(2,7,1));
            subMat2D.push_back(af_make_seq(2,7,1));

            subMat3D.push_back(af_make_seq(2,7,1));
            subMat3D.push_back(af_make_seq(2,7,1));
            subMat3D.push_back(af_span);
        }
        vector<af_seq> subMat2D;
        vector<af_seq> subMat3D;
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Transpose, TestTypes);

template<typename T>
void trsTest(string pTestFile, bool isSubRef=false, const vector<af_seq> *seqv=NULL)
{
    if (noDoubleTests<T>())
        return;

    vector<af::dim4> numDims;

    vector<vector<T> >   in;
    vector<vector<T> >   tests;
    readTests<T,T,int>(pTestFile,numDims,in,tests);
    af::dim4 dims       = numDims[0];

    af_array outArray   = 0;
    af_array inArray    = 0;
    T *outData;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    // check if the test is for indexed Array
    if (isSubRef) {
        af::dim4 newDims(dims[1]-4,dims[0]-4,dims[2],dims[3]);
        af_array subArray = 0;
        ASSERT_EQ(AF_SUCCESS, af_index(&subArray,inArray,seqv->size(),&seqv->front()));
        ASSERT_EQ(AF_SUCCESS, af_transpose(&outArray,subArray, false));
        // destroy the temporary indexed Array
        ASSERT_EQ(AF_SUCCESS, af_release_array(subArray));

        dim_t nElems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems,outArray));
        outData = new T[nElems];
    } else {
        ASSERT_EQ(AF_SUCCESS,af_transpose(&outArray,inArray, false));
        outData = new T[dims.elements()];
    }

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<T> currGoldBar   = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter],outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(Transpose,Vector)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/vector.test"));
}

TYPED_TEST(Transpose,VectorBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/vector_batch.test"));
}

 TYPED_TEST(Transpose,Square)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/square.test"));
}

TYPED_TEST(Transpose,Rectangle)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle.test"));
}

TYPED_TEST(Transpose,Rectangle2)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle2.test"));
}

TYPED_TEST(Transpose,SquareBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/square_batch.test"));
}

TYPED_TEST(Transpose,RectangleBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle_batch.test"));
}

TYPED_TEST(Transpose,RectangleBatch2)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle_batch2.test"));
}

TYPED_TEST(Transpose,Square512x512)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/square2.test"));
}

TYPED_TEST(Transpose,SubRef)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/offset.test"),true,&(this->subMat2D));
}

TYPED_TEST(Transpose,SubRefBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/offset_batch.test"),true,&(this->subMat3D));
}


////////////////////////////////////// CPP //////////////////////////////////
//
template<typename T>
void trsCPPTest(string pFileName)
{
    vector<af::dim4> numDims;

    vector<vector<T> >   in;
    vector<vector<T> >   tests;
    readTests<T, T, int>(pFileName, numDims, in, tests);
    af::dim4 dims = numDims[0];

    if (noDoubleTests<T>()) return;

    af::array input(dims, &(in[0].front()));
    af::array output = af::transpose(input);

    T *outData = new T[dims.elements()];
    output.host((void*)outData);

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter << std::endl;
        }
    }

    // cleanup
    delete[] outData;
}

TEST(Transpose, CPP_f64)
{
    trsCPPTest<double>(string(TEST_DIR"/transpose/rectangle_batch2.test"));
}

TEST(Transpose, CPP_f32)
{
    trsCPPTest<float>(string(TEST_DIR"/transpose/rectangle_batch2.test"));
}

template<typename T>
void trsCPPConjTest()
{
    vector<af::dim4> numDims;

    af::dim4 dims(40, 40);

    if (noDoubleTests<T>()) return;

    af::array input = randu(dims, (af_dtype) af::dtype_traits<T>::af_type);
    af::array output_t = af::transpose(input, false);
    af::array output_c = af::transpose(input, true);

    T *tData  = new T[dims.elements()];
    T *cData = new T[dims.elements()];
    output_t.host((void*)tData);
    output_c.host((void*)cData);

    size_t nElems = dims.elements();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(real(tData[elIter]), real(cData[elIter]), 1e-6)<< "at: " << elIter << std::endl;
        ASSERT_NEAR(-imag(tData[elIter]), imag(cData[elIter]), 1e-6)<< "at: " << elIter << std::endl;
    }

    // cleanup
    delete[] tData;
    delete[] cData;
}

TEST(Transpose, CPP_c32_CONJ)
{
    trsCPPConjTest<cfloat>();
}

TEST(Transpose, GFOR)
{
    using namespace af;
    dim4 dims = dim4(100, 100, 3);
    array A = round(100 * randu(dims));
    array B = constant(0, 100, 100, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = A(span, span, ii).T();
    }

    for(int ii = 0; ii < 3; ii++) {
        array c_ii = A(span, span, ii).T();
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
