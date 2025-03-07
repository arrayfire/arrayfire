/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype_traits;
using af::product;
using af::seq;
using af::span;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Tile : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort,
                         half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Tile, TestTypes);

template<typename T>
void tileTest(string pTestFile, const unsigned resultIdx, const uint x,
              const uint y, const uint z, const uint w, bool isSubRef = false,
              const vector<af_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(af_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(af_tile(&outArray, inArray, x, y, z, w));

    dim4 goldDims(idims[0] * x, idims[1] * y, idims[2] * z, idims[3] * w);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, outArray);

    if (inArray != 0) af_release_array(inArray);
    if (outArray != 0) af_release_array(outArray);
    if (tempArray != 0) af_release_array(tempArray);
}

#define TILE_INIT(desc, file, resultIdx, x, y, z, w)                 \
    TYPED_TEST(Tile, desc) {                                         \
        tileTest<TypeParam>(string(TEST_DIR "/tile/" #file ".test"), \
                            resultIdx, x, y, z, w);                  \
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
TEST(Tile, CPP) {
    const unsigned resultIdx = 0;
    const unsigned x         = 2;
    const unsigned y         = 2;
    const unsigned z         = 2;
    const unsigned w         = 1;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/tile/tile_large3D.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));
    array output = tile(input, x, y, z, w);

    dim4 goldDims(idims[0] * x, idims[1] * y, idims[2] * z, idims[3] * w);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], goldDims, output);
}

TEST(Tile, MaxDim) {
    const size_t largeDim = 65535 * 32 + 1;
    const unsigned x      = 1;
    const unsigned z      = 1;
    unsigned y            = 2;
    unsigned w            = 1;

    array input  = constant(1, 1, largeDim);
    array output = tile(input, x, y, z, w);

    ASSERT_EQ(1, output.dims(0));
    ASSERT_EQ(2 * largeDim, output.dims(1));
    ASSERT_EQ(1, output.dims(2));
    ASSERT_EQ(1, output.dims(3));

    ASSERT_EQ(1.f, product<float>(output));

    y = 1;
    w = 2;

    input  = constant(1, 1, 1, 1, largeDim);
    output = tile(input, x, y, z, w);

    ASSERT_EQ(1, output.dims(0));
    ASSERT_EQ(1, output.dims(1));
    ASSERT_EQ(1, output.dims(2));
    ASSERT_EQ(2 * largeDim, output.dims(3));

    ASSERT_EQ(1.f, product<float>(output));
}

TEST(Tile, DocSnippet) {
    //! [ex_tile_input]
    float hA[] = {0, 1, 2, 3, 4, 5};
    array A(3, 2, hA);
    //  0.  3.
    //  1.  4.
    //  2.  5.
    //! [ex_tile_input]

    //! [ex_tile_0_2]
    array B = tile(A, 2, 1);
    //  0.  3.
    //  1.  4.
    //  2.  5.
    //  0.  3.
    //  1.  4.
    //  2.  5.
    //! [ex_tile_0_2]

    ASSERT_ARRAYS_EQ(A, B(seq(0, 2), span));
    ASSERT_ARRAYS_EQ(A, B(seq(3, 5), span));

    //! [ex_tile_1_3]
    array C = tile(A, 1, 3);
    //  0.  3.  0.  3.  0.  3.
    //  1.  4.  1.  4.  1.  4.
    //  2.  5.  2.  5.  2.  5.
    //! [ex_tile_1_3]

    ASSERT_ARRAYS_EQ(A, C(span, seq(0, 1)));
    ASSERT_ARRAYS_EQ(A, C(span, seq(2, 3)));
    ASSERT_ARRAYS_EQ(A, C(span, seq(4, 5)));

    //! [ex_tile_0_2_and_1_3]
    array D = tile(A, 2, 3);
    //  0.  3.  0.  3.  0.  3.
    //  1.  4.  1.  4.  1.  4.
    //  2.  5.  2.  5.  2.  5.
    //  0.  3.  0.  3.  0.  3.
    //  1.  4.  1.  4.  1.  4.
    //  2.  5.  2.  5.  2.  5.
    //! [ex_tile_0_2_and_1_3]

    ASSERT_ARRAYS_EQ(C, D(seq(0, 2), span));
    ASSERT_ARRAYS_EQ(C, D(seq(3, 5), span));
}

// The tile was failing for larger sizes because of JIT kernels were not able to
// handle repeated x blocks. The kernels were exiting early which caused the
// next iteration to fail
TEST(Tile, LargeRepeatDim) {
    long long dim0     = 33;
    long long largeDim = 40001;
    array temp_ones    = af::iota(largeDim, dim4(1), u8);
    temp_ones          = af::moddims(temp_ones, 1, 1, largeDim);
    temp_ones.eval();

    array temp = tile(temp_ones, dim0, 1, 1);
    temp.eval();
    vector<unsigned char> empty(dim0 * largeDim);
    for (long long ii = 0; ii < largeDim; ii++) {
        int offset = ii * dim0;
        for (int i = 0; i < dim0; i++) { empty[offset + i] = ii; }
    }

    ASSERT_VEC_ARRAY_EQ(empty, dim4(dim0, 1, largeDim), temp);
}
