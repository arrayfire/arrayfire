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
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Rotate : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, intl, char> TestTypes;

// register the type list
TYPED_TEST_CASE(Rotate, TestTypes);

#define PI 3.1415926535897931f

template<typename T>
void rotateTest(string pTestFile, const unsigned resultIdx, const float angle, const bool crop, const bool recenter, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> >   in;
    vector<vector<T> >   tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 dims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    float theta = angle * PI / 180.0f;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_rotate(&outArray, inArray, theta, crop, AF_INTERP_NEAREST));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();

    // This is a temporary solution. The reason we need this is because of
    // floating point error in the index computations on CPU/GPU, some
    // elements of GPU(CUDA/OpenCL) versions are different from the CPU version.
    // That is, the input index of CPU/GPU may differ by 1 (rounding error) on
    // x or y, hence a different value is copied.
    // We expect 99.99% values to be same between the CPU/GPU versions and
    // ASSERT_EQ (in comments below) to pass for CUDA & OpenCL backends
    size_t fail_count = 0;
    for(size_t i = 0; i < nElems; i++) {
        if(abs((tests[resultIdx][i] - (T)outData[i])) > 0.001)
            fail_count++;
    }
    ASSERT_EQ(true, ((fail_count / (float)nElems) < 0.005));

    //for (size_t elIter = 0; elIter < nElems; ++elIter) {
    //    ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    //}


    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define ROTATE_INIT(desc, file, resultIdx, angle, crop, recenter)                               \
    TYPED_TEST(Rotate, desc)                                                                    \
    {                                                                                           \
        rotateTest<TypeParam>(string(TEST_DIR"/rotate/"#file".test"), resultIdx, angle, crop, recenter);\
    }

    ROTATE_INIT(Square180NoCropRecenter     , rotate1,  0, 180, false, true);
    ROTATE_INIT(Square180CropRecenter       , rotate1,  1, 180, true , true);
    ROTATE_INIT(Square90NoCropRecenter      , rotate1,  2, 90 , false, true);
    ROTATE_INIT(Square90CropRecenter        , rotate1,  3, 90 , true , true);
    ROTATE_INIT(Square45NoCropRecenter      , rotate1,  4, 45 , false, true);
    ROTATE_INIT(Square45CropRecenter        , rotate1,  5, 45 , true , true);
    ROTATE_INIT(Squarem45NoCropRecenter     , rotate1,  6,-45 , false, true);
    ROTATE_INIT(Squarem45CropRecenter       , rotate1,  7,-45 , true , true);
    ROTATE_INIT(Square60NoCropRecenter      , rotate1,  8, 60 , false, true);
    ROTATE_INIT(Square60CropRecenter        , rotate1,  9, 60 , true , true);
    ROTATE_INIT(Square30NoCropRecenter      , rotate1, 10, 30 , false, true);
    ROTATE_INIT(Square30CropRecenter        , rotate1, 11, 30 , true , true);
    ROTATE_INIT(Square15NoCropRecenter      , rotate1, 12, 15 , false, true);
    ROTATE_INIT(Square15CropRecenter        , rotate1, 13, 15 , true , true);
    ROTATE_INIT(Square10NoCropRecenter      , rotate1, 14, 10 , false, true);
    ROTATE_INIT(Square10CropRecenter        , rotate1, 15, 10 , true , true);
    ROTATE_INIT(Square01NoCropRecenter      , rotate1, 16,  1 , false, true);
    ROTATE_INIT(Square01CropRecenter        , rotate1, 17,  1 , true , true);
    ROTATE_INIT(Square360NoCropRecenter     , rotate1, 18, 360, false, true);
    ROTATE_INIT(Square360CropRecenter       , rotate1, 19, 360, true , true);
    ROTATE_INIT(Squarem180NoCropRecenter    , rotate1, 20,-180, false, true);
    ROTATE_INIT(Squarem180CropRecenter      , rotate1, 21,-180, false, true);
    ROTATE_INIT(Square00NoCropRecenter      , rotate1, 22,  0 , false, true);
    ROTATE_INIT(Square00CropRecenter        , rotate1, 23,  0 , true , true);

    ROTATE_INIT(Rectangle180NoCropRecenter     , rotate2,  0, 180, false, true);
    ROTATE_INIT(Rectangle180CropRecenter       , rotate2,  1, 180, true , true);
    ROTATE_INIT(Rectangle90NoCropRecenter      , rotate2,  2, 90 , false, true);
    ROTATE_INIT(Rectangle90CropRecenter        , rotate2,  3, 90 , true , true);
    ROTATE_INIT(Rectangle45NoCropRecenter      , rotate2,  4, 45 , false, true);
    ROTATE_INIT(Rectangle45CropRecenter        , rotate2,  5, 45 , true , true);
    ROTATE_INIT(Rectanglem45NoCropRecenter     , rotate2,  6,-45 , false, true);
    ROTATE_INIT(Rectanglem45CropRecenter       , rotate2,  7,-45 , true , true);
    ROTATE_INIT(Rectangle60NoCropRecenter      , rotate2,  8, 60 , false, true);
    ROTATE_INIT(Rectangle60CropRecenter        , rotate2,  9, 60 , true , true);
    ROTATE_INIT(Rectangle30NoCropRecenter      , rotate2, 10, 30 , false, true);
    ROTATE_INIT(Rectangle30CropRecenter        , rotate2, 11, 30 , true , true);
    ROTATE_INIT(Rectangle15NoCropRecenter      , rotate2, 12, 15 , false, true);
    ROTATE_INIT(Rectangle15CropRecenter        , rotate2, 13, 15 , true , true);
    ROTATE_INIT(Rectangle10NoCropRecenter      , rotate2, 14, 10 , false, true);
    ROTATE_INIT(Rectangle10CropRecenter        , rotate2, 15, 10 , true , true);
    ROTATE_INIT(Rectangle01NoCropRecenter      , rotate2, 16,  1 , false, true);
    ROTATE_INIT(Rectangle01CropRecenter        , rotate2, 17,  1 , true , true);
    ROTATE_INIT(Rectangle360NoCropRecenter     , rotate2, 18, 360, false, true);
    ROTATE_INIT(Rectangle360CropRecenter       , rotate2, 19, 360, true , true);
    ROTATE_INIT(Rectanglem180NoCropRecenter    , rotate2, 20,-180, false, true);
    ROTATE_INIT(Rectanglem180CropRecenter      , rotate2, 21,-180, false, true);
    ROTATE_INIT(Rectangle00NoCropRecenter      , rotate2, 22,  0 , false, true);
    ROTATE_INIT(Rectangle00CropRecenter        , rotate2, 23,  0 , true , true);

////////////////////////////////// CPP //////////////////////////////////////
//
TEST(Rotate, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 0;
    const float angle = 180;
    const bool crop = false;

    vector<af::dim4> numDims;
    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(string(TEST_DIR"/rotate/rotate1.test"),numDims,in,tests);

    af::dim4 dims = numDims[0];
    float theta = angle * PI / 180.0f;

    af::array input(dims, &(in[0].front()));
    af::array output = af::rotate(input, theta, crop, AF_INTERP_NEAREST);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();

    // This is a temporary solution. The reason we need this is because of
    // floating point error in the index computations on CPU/GPU, some
    // elements of GPU(CUDA/OpenCL) versions are different from the CPU version.
    // That is, the input index of CPU/GPU may differ by 1 (rounding error) on
    // x or y, hence a different value is copied.
    // We expect 99.99% values to be same between the CPU/GPU versions and
    // ASSERT_EQ (in comments below) to pass for CUDA & OpenCL backends
    size_t fail_count = 0;
    for(size_t i = 0; i < nElems; i++) {
        if(fabs(tests[resultIdx][i] - outData[i]) > 0.0001)
            fail_count++;
    }
    ASSERT_EQ(true, ((fail_count / (float)nElems) < 0.01));

    // Delete
    delete[] outData;
}
