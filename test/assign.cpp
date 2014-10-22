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
            subMat1D.push_back({5,20,1});

            subMat2D.push_back({1,2,1});
            subMat2D.push_back({1,2,1});

            subMat3D.push_back({3,4,1});
            subMat3D.push_back({0,1,1});
            subMat3D.push_back({1,2,1});

            subMat4D.push_back({3,4,1});
            subMat4D.push_back({0,1,1});
            subMat4D.push_back({0,1,1});
            subMat4D.push_back({1,2,1});

            subMat1D_to_2D.push_back({1,2,1});
            subMat1D_to_2D.push_back({1,1,1});

            subMat1D_to_3D.push_back({5,20,1});
            subMat1D_to_3D.push_back({1,1,1});
            subMat1D_to_3D.push_back({2,2,1});

            subMat2D_to_3D.push_back({3,4,1});
            subMat2D_to_3D.push_back({0,1,1});
            subMat2D_to_3D.push_back({1,1,1});

            subMat1D_to_4D.push_back({3,4,1});
            subMat1D_to_4D.push_back({0,0,1});
            subMat1D_to_4D.push_back({0,0,1});
            subMat1D_to_4D.push_back({1,1,1});

            subMat2D_to_4D.push_back({3,4,1});
            subMat2D_to_4D.push_back({0,1,1});
            subMat2D_to_4D.push_back({0,0,1});
            subMat2D_to_4D.push_back({1,1,1});

            subMat3D_to_4D.push_back({3,4,1});
            subMat3D_to_4D.push_back({0,1,1});
            subMat3D_to_4D.push_back({0,1,1});
            subMat3D_to_4D.push_back({1,1,1});
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
typedef ::testing::Types<af::af_cdouble, af::af_cfloat, double, float, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(ArrayAssign, TestTypes);

template<typename inType, typename outType>
void assignTest(string pTestFile, const vector<af_seq> *seqv)
{
    vector<af::dim4>  numDims;
    vector<vector<inType>>      in;
    vector<vector<outType>>   tests;

    readTests<inType, outType, int>(pTestFile, numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af_array outArray  = 0;
    af_array inArray   = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&outArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<outType>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_assign(outArray, seqv->size(), &seqv->front(), inArray));

    outType *outData = new outType[dims1.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    vector<outType> currGoldBar = tests[0];
    size_t nElems        = currGoldBar.size();
    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
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
    vector<af::af_cfloat> in(10, af::af_cfloat(0,0));
    vector<float> tests(100, float(1));

    af::dim4 dims0(10, 1, 1, 1);
    af::dim4 dims1(100, 1, 1, 1);
    af_array outArray  = 0;
    af_array inArray   = 0;

    vector<af_seq> seqv;
    seqv.push_back({5,14,1});

    ASSERT_EQ(AF_ERR_ARG, af_assign(outArray, seqv.size(), &seqv.front(), inArray));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in.front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<af::af_cfloat>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_assign(outArray, seqv.size(), &seqv.front(), inArray));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&outArray, &(in.front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_assign(outArray, 0, &seqv.front(), inArray));

    ASSERT_EQ(AF_ERR_INVALID_TYPE, af_assign(outArray, seqv.size(), &seqv.front(), inArray));

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}
