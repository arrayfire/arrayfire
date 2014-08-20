#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <testHelpers.hpp>

using std::string;
using std::cout;
using std::endl;
using std::vector;
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Transpose : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat2D.push_back({2,7,1});
            subMat2D.push_back({2,7,1});

            subMat3D.push_back({2,7,1});
            subMat3D.push_back({2,7,1});
            subMat3D.push_back(span);
        }
        vector<af_seq> subMat2D;
        vector<af_seq> subMat3D;
};

// create a list of types to be tested
typedef ::testing::Types<float, af_cfloat, double, af_cdouble, int, unsigned, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Transpose, TestTypes);

template<typename T>
void trsTest(string pTestFile, bool isSubRef=false, const vector<af_seq> *seqv=nullptr)
{
    af::dim4            dims(1);
    vector<T>           in;
    vector<vector<T>>   tests;
    ReadTests<int, T>(pTestFile,dims,in,tests);

    af_array outArray   = 0;
    af_array inArray    = 0;
    T *outData;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    // check if the test is for indexed Array
    if (isSubRef) {
        af::dim4 newDims(dims[1]-4,dims[0]-4,dims[2],dims[3]);
        af_array subArray = 0;
        ASSERT_EQ(AF_SUCCESS, af_index(&subArray,inArray,seqv->size(),&seqv->front()));
        ASSERT_EQ(AF_SUCCESS, af_transpose(&outArray,subArray));
        // destroy the temporary indexed Array
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(subArray));

        dim_type nElems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems,outArray));
        outData = new T[nElems];
    } else {
        ASSERT_EQ(AF_SUCCESS, af_transpose(&outArray,inArray));
        outData = new T[dims.elements()];
    }

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (int testIter=0; testIter<tests.size(); ++testIter) {
        vector<T> currGoldBar   = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter],outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
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

TYPED_TEST(Transpose,SquareBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/square_batch.test"));
}


TYPED_TEST(Transpose,Rectangle)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle.test"));
}

TYPED_TEST(Transpose,RectangleBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/rectangle_batch.test"));
}

TYPED_TEST(Transpose,SubRef)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/offset.test"),true,&(this->subMat2D));
}

TYPED_TEST(Transpose,SubRefBatch)
{
    trsTest<TypeParam>(string(TEST_DIR"/transpose/offset_batch.test"),true,&(this->subMat3D));
}

TYPED_TEST(Transpose,InvalidArgs)
{
    af::dim4            dims(1);
    vector<TypeParam>           in;
    vector<vector<TypeParam>>   tests;
    ReadTests<int, TypeParam>(string(TEST_DIR"/transpose/square.test"),dims,in,tests);

    af_array inArray   = 0;
    af_array outArray  = 0;

    // square test file is 100x100 originally
    // usee new dimensions for this argument
    // unit test
    af::dim4 newDims(5,5,2,2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), newDims.ndims(), newDims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_transpose(&outArray,inArray));
}
