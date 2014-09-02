#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/blas.h>
#include <af/traits.hpp>
#include <af/defines.h>
#include <testHelpers.hpp>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::ostream_iterator;
using std::copy;
using std::vector;

template<typename T>
class MatrixMultiply : public ::testing::Test
{

};

typedef ::testing::Types<float, af::af_cfloat, double, af::af_cdouble> TestTypes;
TYPED_TEST_CASE(MatrixMultiply, TestTypes);

template<typename T, bool isBVector = false>
void
MatMulCheck(string TestFile)
{
    using std::vector;
    vector<af::dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);

    af_array a, aT, b, bT;
    ASSERT_EQ(AF_SUCCESS,
            af_create_array(&a, &hData[0].front(), numDims[0].ndims(), numDims[0].get(), (af_dtype) af::dtype_traits<T>::af_type));
    af::dim4 atdims = numDims[0];
    {
        dim_type f  =    atdims[0];
        atdims[0]   =    atdims[1];
        atdims[1]   =    f;
    }
    ASSERT_EQ(AF_SUCCESS,
            af_moddims(&aT, a, atdims.ndims(), atdims.get()));
    ASSERT_EQ(AF_SUCCESS,
            af_create_array(&b, &hData[1].front(), numDims[1].ndims(), numDims[1].get(), (af_dtype) af::dtype_traits<T>::af_type));
    af::dim4 btdims = numDims[1];
    {
        dim_type f = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }
    ASSERT_EQ(AF_SUCCESS,
            af_moddims(&bT, b, btdims.ndims(), btdims.get()));

    vector<af_array> out(tests.size(), 0);
    if(isBVector) {
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[0] , aT, b,    AF_NO_TRANSPOSE,    AF_NO_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[1] , bT, a,   AF_NO_TRANSPOSE,    AF_NO_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[2] , b, a,    AF_TRANSPOSE,       AF_NO_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[3] , bT, aT,   AF_NO_TRANSPOSE,    AF_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[4] , b, aT,    AF_TRANSPOSE,       AF_TRANSPOSE));
    }
    else {
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[0] , a, b, AF_NO_TRANSPOSE,   AF_NO_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[1] , a, bT, AF_NO_TRANSPOSE,   AF_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[2] , a, bT, AF_TRANSPOSE,      AF_NO_TRANSPOSE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[3] , aT, bT, AF_TRANSPOSE,      AF_TRANSPOSE));
    }

    for(size_t i = 0; i < tests.size(); i++) {
        dim_type elems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&elems, out[i]));
        vector<T> h_out(elems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void *)&h_out.front(), out[i]));

        if( false == equal(h_out.begin(), h_out.end(), tests[i].begin()) ) {

            cout << "Failed test " << i << "\nCalculated: " << endl;
            cout << "Expected: " << endl;
            copy(tests[i].begin(), tests[i].end(), ostream_iterator<T>(cout, ", "));
            FAIL();
        }
    }
}

TYPED_TEST(MatrixMultiply, Square)
{
    MatMulCheck<TypeParam>(TEST_DIR"/blas/Basic.test");
}

TYPED_TEST(MatrixMultiply, NonSquare)
{
    MatMulCheck<TypeParam>(TEST_DIR"/blas/NonSquare.test");
}

TYPED_TEST(MatrixMultiply, SquareVector)
{
    MatMulCheck<TypeParam, true>(TEST_DIR"/blas/SquareVector.test");
}

TYPED_TEST(MatrixMultiply, RectangleVector)
{
    MatMulCheck<TypeParam, true>(TEST_DIR"/blas/RectangleVector.test");
}

