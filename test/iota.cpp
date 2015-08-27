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
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Iota : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned int, intl, uintl, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Iota, TestTypes);

template<typename T>
void iotaTest(const af::dim4 idims, const af::dim4 tdims)
{
    if (noDoubleTests<T>()) return;

    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_iota(&outArray, idims.ndims(), idims.get(),
               tdims.ndims(), tdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    af_array temp0 = 0, temp1 = 0, temp2 = 0;
    af::dim4 tempdims(idims.elements());
    af::dim4 fulldims;
    for(unsigned i = 0; i < 4; i++) {
        fulldims[i] = idims[i] * tdims[i];
    }
    ASSERT_EQ(AF_SUCCESS, af_range(&temp2, tempdims.ndims(), tempdims.get(), 0, (af_dtype) af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_moddims(&temp1, temp2, idims.ndims(), idims.get()));
    ASSERT_EQ(AF_SUCCESS, af_tile(&temp0, temp1, tdims[0], tdims[1], tdims[2], tdims[3]));

    // Get result
    T* outData = new T[fulldims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    T* tileData = new T[fulldims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)tileData, temp0));

    // Compare result
    for(int i = 0; i < (int) fulldims.elements(); i++)
        ASSERT_EQ(tileData[i], outData[i]) << "at: " << i << std::endl;

    // Delete
    delete[] outData;
    delete[] tileData;

    if(outArray  != 0) af_release_array(outArray);
    if(temp0     != 0) af_release_array(temp0);
    if(temp1     != 0) af_release_array(temp1);
    if(temp2     != 0) af_release_array(temp2);
}

#define IOTA_INIT(desc, x, y, z, w, a, b, c, d)                                             \
    TYPED_TEST(Iota, desc)                                                                  \
    {                                                                                       \
        iotaTest<TypeParam>(af::dim4(x, y, z, w), af::dim4(a, b, c, d));                    \
    }

    IOTA_INIT(Iota1D0, 100,  1, 1, 1, 2, 3, 1, 1);

    IOTA_INIT(Iota2D0,  10, 20, 1, 1, 3, 1, 2, 1);
    IOTA_INIT(Iota2D1, 100,  5, 1, 1, 1, 2, 4, 2);

    IOTA_INIT(Iota3D0,  20,  6, 3, 1, 1, 1, 1, 1);
    IOTA_INIT(Iota3D1,  10, 12, 5, 1, 2, 3, 4, 5);
    IOTA_INIT(Iota3D2,  25, 30, 2, 1, 1, 2, 2, 1);

    IOTA_INIT(Iota4D0,  20,  6, 3, 2, 2, 3, 1, 2);
    IOTA_INIT(Iota4D1,  10, 12, 5, 2, 1, 2, 2, 2);
    IOTA_INIT(Iota4D2,  25, 30, 2, 2, 3, 2, 1, 1);
    IOTA_INIT(Iota4D3,  25, 30, 2, 2, 4, 2, 4, 2);

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Iota, CPP)
{
    if (noDoubleTests<float>()) return;

    af::dim4 idims(23, 15, 1, 1);
    af::dim4 tdims(2, 2, 1, 1);
    af::dim4 fulldims;
    for(unsigned i = 0; i < 4; i++) {
        fulldims[i] = idims[i] * tdims[i];
    }

    af::array output = af::iota(idims, tdims);
    af::array tileArray = af::tile(af::moddims(af::range(af::dim4(idims.elements()), 0), idims), tdims);

    // Get result
    float* outData = new float[fulldims.elements()];
    output.host((void*)outData);

    float* tileData = new float[fulldims.elements()];
    tileArray.host((void*)tileData);

    // Compare result

    // Compare result
    for(int i = 0; i < (int)fulldims.elements(); i++)
        ASSERT_EQ(tileData[i], outData[i]) << "at: " << i << std::endl;

    // Delete
    delete[] outData;
    delete[] tileData;
}
