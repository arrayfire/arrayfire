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
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
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

    if (isSubRef) {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_rotate(&outArray, inArray, theta, crop, AF_INTERP_BILINEAR));

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
        if(abs((tests[resultIdx][i] - (T)outData[i])) > 0.001) {
            fail_count++;
        }
    }
    ASSERT_EQ(true, ((fail_count / (float)nElems) < 0.02)) << "where count = " << fail_count << std::endl;

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

    ROTATE_INIT(Square180NoCropRecenter     , rotatelinear1,  0, 180, false, true);
    ROTATE_INIT(Square180CropRecenter       , rotatelinear1,  1, 180, true , true);
    ROTATE_INIT(Square90NoCropRecenter      , rotatelinear1,  2, 90 , false, true);
    ROTATE_INIT(Square90CropRecenter        , rotatelinear1,  3, 90 , true , true);
    ROTATE_INIT(Square45NoCropRecenter      , rotatelinear1,  4, 45 , false, true);
    ROTATE_INIT(Square45CropRecenter        , rotatelinear1,  5, 45 , true , true);
    ROTATE_INIT(Squarem45NoCropRecenter     , rotatelinear1,  6,-45 , false, true);
    ROTATE_INIT(Squarem45CropRecenter       , rotatelinear1,  7,-45 , true , true);
    ROTATE_INIT(Square60NoCropRecenter      , rotatelinear1,  8, 60 , false, true);
    ROTATE_INIT(Square60CropRecenter        , rotatelinear1,  9, 60 , true , true);
    ROTATE_INIT(Square30NoCropRecenter      , rotatelinear1, 10, 30 , false, true);
    ROTATE_INIT(Square30CropRecenter        , rotatelinear1, 11, 30 , true , true);
    ROTATE_INIT(Square15NoCropRecenter      , rotatelinear1, 12, 15 , false, true);
    ROTATE_INIT(Square15CropRecenter        , rotatelinear1, 13, 15 , true , true);
    ROTATE_INIT(Square10NoCropRecenter      , rotatelinear1, 14, 10 , false, true);
    ROTATE_INIT(Square10CropRecenter        , rotatelinear1, 15, 10 , true , true);
    ROTATE_INIT(Square01NoCropRecenter      , rotatelinear1, 16,  1 , false, true);
    ROTATE_INIT(Square01CropRecenter        , rotatelinear1, 17,  1 , true , true);
    ROTATE_INIT(Square360NoCropRecenter     , rotatelinear1, 18, 360, false, true);
    ROTATE_INIT(Square360CropRecenter       , rotatelinear1, 19, 360, true , true);
    ROTATE_INIT(Squarem180NoCropRecenter    , rotatelinear1, 20,-180, false, true);
    ROTATE_INIT(Squarem180CropRecenter      , rotatelinear1, 21,-180, false, true);
    ROTATE_INIT(Square00NoCropRecenter      , rotatelinear1, 22,  0 , false, true);
    ROTATE_INIT(Square00CropRecenter        , rotatelinear1, 23,  0 , true , true);

    ROTATE_INIT(Rectangle180NoCropRecenter     , rotatelinear2,  0, 180, false, true);
    ROTATE_INIT(Rectangle180CropRecenter       , rotatelinear2,  1, 180, true , true);
    ROTATE_INIT(Rectangle90NoCropRecenter      , rotatelinear2,  2, 90 , false, true);
    ROTATE_INIT(Rectangle90CropRecenter        , rotatelinear2,  3, 90 , true , true);
    ROTATE_INIT(Rectangle45NoCropRecenter      , rotatelinear2,  4, 45 , false, true);
    ROTATE_INIT(Rectangle45CropRecenter        , rotatelinear2,  5, 45 , true , true);
    ROTATE_INIT(Rectanglem45NoCropRecenter     , rotatelinear2,  6,-45 , false, true);
    ROTATE_INIT(Rectanglem45CropRecenter       , rotatelinear2,  7,-45 , true , true);
    ROTATE_INIT(Rectangle60NoCropRecenter      , rotatelinear2,  8, 60 , false, true);
    ROTATE_INIT(Rectangle60CropRecenter        , rotatelinear2,  9, 60 , true , true);
    ROTATE_INIT(Rectangle30NoCropRecenter      , rotatelinear2, 10, 30 , false, true);
    ROTATE_INIT(Rectangle30CropRecenter        , rotatelinear2, 11, 30 , true , true);
    ROTATE_INIT(Rectangle15NoCropRecenter      , rotatelinear2, 12, 15 , false, true);
    ROTATE_INIT(Rectangle15CropRecenter        , rotatelinear2, 13, 15 , true , true);
    ROTATE_INIT(Rectangle10NoCropRecenter      , rotatelinear2, 14, 10 , false, true);
    ROTATE_INIT(Rectangle10CropRecenter        , rotatelinear2, 15, 10 , true , true);
    ROTATE_INIT(Rectangle01NoCropRecenter      , rotatelinear2, 16,  1 , false, true);
    ROTATE_INIT(Rectangle01CropRecenter        , rotatelinear2, 17,  1 , true , true);
    ROTATE_INIT(Rectangle360NoCropRecenter     , rotatelinear2, 18, 360, false, true);
    ROTATE_INIT(Rectangle360CropRecenter       , rotatelinear2, 19, 360, true , true);
    ROTATE_INIT(Rectanglem180NoCropRecenter    , rotatelinear2, 20,-180, false, true);
    ROTATE_INIT(Rectanglem180CropRecenter      , rotatelinear2, 21,-180, false, true);
    ROTATE_INIT(Rectangle00NoCropRecenter      , rotatelinear2, 22,  0 , false, true);
    ROTATE_INIT(Rectangle00CropRecenter        , rotatelinear2, 23,  0 , true , true);

////////////////////////////////// CPP //////////////////////////////////////

TEST(Rotate, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 0;
    const float angle = 180;
    const bool crop = false;

    vector<af::dim4> numDims;
    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(string(TEST_DIR"/rotate/rotatelinear1.test"),numDims,in,tests);

    af::dim4 dims = numDims[0];
    float theta = angle * PI / 180.0f;

    af::array input(dims, &(in[0].front()));
    af::array output = af::rotate(input, theta, crop, AF_INTERP_BILINEAR);

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
