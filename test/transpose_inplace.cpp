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
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Transpose, TestTypes);

template<typename T>
void transposeip_test(af::dim4 dims)
{
    if (noDoubleTests<T>())
        return;

    af_array inArray  = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_randu(&inArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_transpose(&outArray, inArray, false));
    ASSERT_EQ(AF_SUCCESS, af_transpose_inplace(inArray, false));

    T *outData = new T[dims.elements()];
    T *trsData = new T[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)trsData, inArray));

    dim_t nElems = dims.elements();
    for (int elIter = 0; elIter < (int)nElems; ++elIter) {
        ASSERT_EQ(trsData[elIter] , outData[elIter])<< "at: " << elIter << std::endl;
    }

    // cleanup
    delete[] outData;
    delete[] trsData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

#define INIT_TEST(Side, D3, D4)                                                     \
    TYPED_TEST(Transpose, TranposeIP_##Side)                                        \
    {                                                                               \
        transposeip_test<TypeParam>(af::dim4(Side, Side, D3, D4));                  \
    }

INIT_TEST(10, 1, 1);
INIT_TEST(64, 1, 1);
INIT_TEST(300, 1, 1);
INIT_TEST(1000, 1, 1);
INIT_TEST(100, 2, 1);
INIT_TEST(25, 2, 2);

////////////////////////////////////// CPP //////////////////////////////////
//
void transposeInPlaceCPPTest()
{
    if (noDoubleTests<float>()) return;

    af::dim4 dims(64, 64, 1,1);

    af::array input = randu(dims);
    af::array output = af::transpose(input);
    transposeInPlace(input);

    float *outData = new float[dims.elements()];
    float *trsData = new float[dims.elements()];

    output.host((void*)outData);
    input.host((void*)trsData);

    dim_t nElems = dims.elements();
    for (int elIter = 0; elIter < (int)nElems; ++elIter) {
        ASSERT_EQ(trsData[elIter], outData[elIter])<< "at: " << elIter << std::endl;
    }

    // cleanup
    delete[] outData;
}
