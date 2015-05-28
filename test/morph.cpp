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
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class Morph : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Morph, TestTypes);

template<typename inType, bool isDilation, bool isVolume>
void morphTest(string pTestFile)
{
    if (noDoubleTests<inType>()) return;

    vector<af::dim4>       numDims;
    vector<vector<inType> >      in;
    vector<vector<inType> >   tests;

    readTests<inType,inType,int>(pTestFile, numDims, in, tests);

    af::dim4 dims      = numDims[0];
    af::dim4 maskDims  = numDims[1];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array maskArray = 0;
    inType *outData;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims.ndims(), dims.get(), (af_dtype)af::dtype_traits<inType>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &(in[1].front()),
                maskDims.ndims(), maskDims.get(), (af_dtype)af::dtype_traits<inType>::af_type));

    if (isDilation) {
        if (isVolume)
            ASSERT_EQ(AF_SUCCESS, af_dilate3(&outArray, inArray, maskArray));
        else
            ASSERT_EQ(AF_SUCCESS, af_dilate(&outArray, inArray, maskArray));
    }
    else {
        if (isVolume)
            ASSERT_EQ(AF_SUCCESS, af_erode3(&outArray, inArray, maskArray));
        else
            ASSERT_EQ(AF_SUCCESS, af_erode(&outArray, inArray, maskArray));
    }

    outData = new inType[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<inType> currGoldBar = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(Morph, Dilate3x3)
{
    morphTest<TypeParam, true, false>(string(TEST_DIR"/morph/dilate3x3.test"));
}

TYPED_TEST(Morph, Erode3x3)
{
    morphTest<TypeParam, false, false>(string(TEST_DIR"/morph/erode3x3.test"));
}

TYPED_TEST(Morph, Dilate3x3_Batch)
{
    morphTest<TypeParam, true, false>(string(TEST_DIR"/morph/dilate3x3_batch.test"));
}

TYPED_TEST(Morph, Erode3x3_Batch)
{
    morphTest<TypeParam, false, false>(string(TEST_DIR"/morph/erode3x3_batch.test"));
}

TYPED_TEST(Morph, Dilate3x3x3)
{
    morphTest<TypeParam, true, true>(string(TEST_DIR"/morph/dilate3x3x3.test"));
}

TYPED_TEST(Morph, Erode3x3x3)
{
    morphTest<TypeParam, false, true>(string(TEST_DIR"/morph/erode3x3x3.test"));
}

template<typename T, bool isDilation, bool isColor>
void morphImageTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray  = 0;
        af_array maskArray= 0;
        af_array outArray = 0;
        af_array goldArray= 0;
        dim_t nElems   = 0;

        inFiles[testId].insert(0,string(TEST_DIR"/morph/"));
        outFiles[testId].insert(0,string(TEST_DIR"/morph/"));

        dim4 mdims(3,3,1,1);
        ASSERT_EQ(AF_SUCCESS, af_constant(&maskArray, 1.0,
                    mdims.ndims(), mdims.get(), (af_dtype)af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        if (isDilation)
            ASSERT_EQ(AF_SUCCESS, af_dilate(&outArray, inArray, maskArray));
        else
            ASSERT_EQ(AF_SUCCESS, af_erode(&outArray, inArray, maskArray));

        T * outData = new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        T * goldData= new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.018f));

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));
    }
}

TEST(Morph, Grayscale)
{
    morphImageTest<float, true, false>(string(TEST_DIR"/morph/gray.test"));
}

TEST(Morph, ColorImage)
{
    morphImageTest<float, false, true>(string(TEST_DIR"/morph/color.test"));
}

template<typename T, bool isDilation>
void morphInputTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T>   in(100,1);
    vector<T>   mask(9,1);

    // Check for 1D inputs
    af::dim4 dims = af::dim4(100,1,1,1);
    af::dim4 mdims(3,3,1,1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &mask.front(),
                mdims.ndims(), mdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));
}

TYPED_TEST(Morph, DilateInvalidInput)
{
    morphInputTest<TypeParam,true>();
}

TYPED_TEST(Morph, ErodeInvalidInput)
{
    morphInputTest<TypeParam,false>();
}

template<typename T, bool isDilation>
void morphMaskTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T>   in(100,1);
    vector<T>   mask(16,1);

    // Check for 4D mask
    af::dim4 dims(10,10,1,1);
    af::dim4 mdims(2,2,2,2);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &mask.front(),
                mdims.ndims(), mdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));

    // Check for 1D mask
    mdims = af::dim4(16,1,1,1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &mask.front(),
                mdims.ndims(), mdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TYPED_TEST(Morph, DilateInvalidMask)
{
    morphMaskTest<TypeParam,true>();
}

TYPED_TEST(Morph, ErodeInvalidMask)
{
    morphMaskTest<TypeParam,false>();
}

template<typename T, bool isDilation>
void morph3DMaskTest(void)
{
    if (noDoubleTests<T>()) return;

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T>   in(1000,1);
    vector<T>   mask(81,1);

    // Check for 2D mask
    af::dim4 dims(10,10,10,1);
    af::dim4 mdims(9,9,1,1);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &mask.front(),
                mdims.ndims(), mdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode3(&outArray, inArray, maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));

    // Check for 4D mask
    mdims = af::dim4(3,3,3,3);

    ASSERT_EQ(AF_SUCCESS, af_create_array(&maskArray, &mask.front(),
                mdims.ndims(), mdims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode3(&outArray, inArray, maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(maskArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

TYPED_TEST(Morph, DilateVolumeInvalidMask)
{
    morph3DMaskTest<TypeParam,true>();
}

TYPED_TEST(Morph, ErodeVolumeInvalidMask)
{
    morph3DMaskTest<TypeParam,false>();
}


////////////////////////////////////// CPP //////////////////////////////////
//
template<typename T, bool isDilation, bool isColor>
void cppMorphImageTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        inFiles[testId].insert(0,string(TEST_DIR"/morph/"));
        outFiles[testId].insert(0,string(TEST_DIR"/morph/"));

        af::array mask = af::constant(1.0, 3, 3);
        af::array img = af::loadImage(inFiles[testId].c_str(), isColor);
        af::array gold = af::loadImage(outFiles[testId].c_str(), isColor);
        dim_t nElems   = gold.elements();
        af::array output;

        if (isDilation)
            output = dilate(img, mask);
        else
            output = erode(img, mask);

        T * outData = new T[nElems];
        output.host((void*)outData);

        T * goldData= new T[nElems];
        gold.host((void*)goldData);

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.018f));
        //cleanup
        delete[] outData;
        delete[] goldData;
    }
}

TEST(Morph, Grayscale_CPP)
{
    cppMorphImageTest<float, true, false>(string(TEST_DIR"/morph/gray.test"));
}

TEST(Morph, ColorImage_CPP)
{
    cppMorphImageTest<float, false, true>(string(TEST_DIR"/morph/color.test"));
}

using namespace af;
TEST(Morph, GFOR)
{
    dim4 dims = dim4(10, 10, 3);
    array A = iota(dims);
    array B = constant(0, dims);
    array mask = randu(3,3) > 0.3;

    gfor(seq ii, 3) {
        B(span, span, ii) = erode(A(span, span, ii), mask);
    }

    for(int ii = 0; ii < 3; ii++) {
        array c_ii = erode(A(span, span, ii), mask);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
