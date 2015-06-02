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
class Tile : public ::testing::Test
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
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Tile, TestTypes);

template<typename T>
void tileTest(string pTestFile, const unsigned resultIdx, const uint x, const uint y, const uint z, const uint w,
              bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;
    readTests<T, T, int>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array outArray = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_tile(&outArray, inArray, x, y, z, w));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_release_array(inArray);
    if(outArray  != 0) af_release_array(outArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define TILE_INIT(desc, file, resultIdx, x, y, z, w)                                        \
    TYPED_TEST(Tile, desc)                                                                  \
    {                                                                                       \
        tileTest<TypeParam>(string(TEST_DIR"/tile/"#file".test"), resultIdx, x, y, z, w);   \
    }

    TILE_INIT(Tile432, tile, 0, 4, 3, 2, 1);
    TILE_INIT(Tile111, tile, 1, 1, 1, 1, 1);
    TILE_INIT(Tile123, tile, 2, 1, 2, 3, 1);
    TILE_INIT(Tile312, tile, 3, 3, 1, 2, 1);
    TILE_INIT(Tile231, tile, 4, 2, 3, 1, 1);

    TILE_INIT(Tile3D432, tile_large3D, 0, 2, 2, 2, 1);
    TILE_INIT(Tile3D111, tile_large3D, 1, 1, 1, 1, 1);
    TILE_INIT(Tile3D123, tile_large3D, 2, 1, 2, 3, 1);
    TILE_INIT(Tile3D312, tile_large3D, 3, 3, 1, 2, 1);
    TILE_INIT(Tile3D231, tile_large3D, 4, 2, 3, 1, 1);

    TILE_INIT(Tile2D432, tile_large2D, 0, 2, 2, 2, 1);
    TILE_INIT(Tile2D111, tile_large2D, 1, 1, 1, 1, 1);
    TILE_INIT(Tile2D123, tile_large2D, 2, 1, 2, 3, 1);
    TILE_INIT(Tile2D312, tile_large2D, 3, 3, 1, 2, 1);
    TILE_INIT(Tile2D231, tile_large2D, 4, 2, 3, 1, 1);


///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Tile, CPP)
{
    if (noDoubleTests<float>()) return;

    const unsigned resultIdx = 0;
    const unsigned x = 2;
    const unsigned y = 2;
    const unsigned z = 2;
    const unsigned w = 1;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/tile/tile_large3D.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output = af::tile(input, x, y, z, w);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx][elIter], outData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] outData;
}

