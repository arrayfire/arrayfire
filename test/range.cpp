/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <half.hpp>
#include <testHelpers.hpp>
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
using af::dim4;
using af::dtype_traits;
using af::range;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Range : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

template<typename T>
class RangeMax : public Range<T> {};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned int, intl, uintl,
                         unsigned char, short, ushort, half_float::half>
    AllTypes;

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned int, intl, uintl,
                         unsigned char, short, ushort>
    RegularTypes;

// register the type list
TYPED_TEST_SUITE(Range, AllTypes);
TYPED_TEST_SUITE(RangeMax, RegularTypes);

template<typename T>
void rangeTest(const uint x, const uint y, const uint z, const uint w,
               const uint dim) {
    SUPPORTED_TYPE_CHECK(T);

    dim4 idims(x, y, z, w);

    af_array outArray = 0;

    ASSERT_SUCCESS(af_range(&outArray, idims.ndims(), idims.get(), dim,
                            (af_dtype)dtype_traits<T>::af_type));

    // Get result
    T* outData = new T[idims.elements()];
    ASSERT_SUCCESS(af_get_data_ptr((void*)outData, outArray));

    // Compare result
    for (int w = 0; w < (int)idims[3]; w++) {
        for (int z = 0; z < (int)idims[2]; z++) {
            for (int y = 0; y < (int)idims[1]; y++) {
                for (int x = 0; x < (int)idims[0]; x++) {
                    T val(0);
                    if (dim == 0) {
                        val = x;
                    } else if (dim == 1) {
                        val = y;
                    } else if (dim == 2) {
                        val = z;
                    } else if (dim == 3) {
                        val = w;
                    }
                    dim_t idx = w * idims[0] * idims[1] * idims[2] +
                                z * idims[0] * idims[1] + y * idims[0] + x;

                    ASSERT_EQ(val, outData[idx]) << "at: " << idx;
                }
            }
        }
    }

    // Delete
    delete[] outData;

    if (outArray != 0) af_release_array(outArray);
}

#define RANGE_INIT(desc, x, y, z, w, rep) \
    TYPED_TEST(Range, desc) { rangeTest<TypeParam>(x, y, z, w, rep); }

RANGE_INIT(Range1D0, 100, 1, 1, 1, 0);

RANGE_INIT(Range2D0, 10, 20, 1, 1, 0);
RANGE_INIT(Range2D1, 100, 5, 1, 1, 1);

RANGE_INIT(Range3D0, 20, 6, 3, 1, 0);
RANGE_INIT(Range3D1, 10, 12, 5, 1, 1);
RANGE_INIT(Range3D2, 25, 30, 2, 1, 2);

RANGE_INIT(Range4D0, 20, 6, 3, 2, 0);
RANGE_INIT(Range4D1, 10, 12, 5, 2, 1);
RANGE_INIT(Range4D2, 25, 30, 2, 2, 2);
RANGE_INIT(Range4D3, 25, 30, 2, 2, 3);

#define RANGE_MAX_INIT(desc, x, y, z, w, rep) \
    TYPED_TEST(RangeMax, desc) { rangeTest<TypeParam>(x, y, z, w, rep); }

RANGE_MAX_INIT(Range1DMaxDim0, 65535 * 32 + 1, 1, 1, 1, 0);
RANGE_MAX_INIT(Range1DMaxDim1, 1, 65535 * 32 + 1, 1, 1, 0);
RANGE_MAX_INIT(Range1DMaxDim2, 1, 1, 65535 * 32 + 1, 1, 0);
RANGE_MAX_INIT(Range1DMaxDim3, 1, 1, 1, 65535 * 32 + 1, 0);

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Range, CPP) {
    const unsigned x   = 23;
    const unsigned y   = 15;
    const unsigned z   = 4;
    const unsigned w   = 2;
    const unsigned dim = 2;

    dim4 idims(x, y, z, w);
    array output = range(x, y, z, w, dim, f32);

    // Get result
    float* outData = new float[idims.elements()];
    output.host((void*)outData);

    // Compare result
    for (int w = 0; w < (int)idims[3]; w++) {
        for (int z = 0; z < (int)idims[2]; z++) {
            for (int y = 0; y < (int)idims[1]; y++) {
                for (int x = 0; x < (int)idims[0]; x++) {
                    float val = 0;
                    if (dim == 0) {
                        val = x;
                    } else if (dim == 1) {
                        val = y;
                    } else if (dim == 2) {
                        val = z;
                    } else if (dim == 3) {
                        val = w;
                    }
                    dim_t idx = (w * idims[0] * idims[1] * idims[2]) +
                                (z * idims[0] * idims[1]) + (y * idims[0]) + x;
                    ASSERT_EQ(val, outData[idx]) << "at: " << idx << endl;
                }
            }
        }
    }

    // Delete
    delete[] outData;
}

TEST(Range, SNIPPET_data_func_range) {
    // clang-format off
    //! [ex_data_func_range]
    //!
    // Generates an array of [0, 4] along first dimension
    array a = range(dim4(5));          // a = [0,
                                       //      1,
                                       //      2,
                                       //      3,
                                       //      4]

    // Generates an array of [0, 4] along first dimension, tiled along second dimension
    array b = range(dim4(5, 2));       // b = [0, 0,
                                       //      1, 1,
                                       //      2, 2,
                                       //      3, 3,
                                       //      4, 4]

    // Generates an array of [0, 2] along second dimension, tiled along first dimension
    array c = range(dim4(5, 3), 1);    // c = [0, 1, 2,
                                       //      0, 1, 2,
                                       //      0, 1, 2,
                                       //      0, 1, 2,
                                       //      0, 1, 2]

    //! [ex_data_func_range]
    // clang-format on

    using std::vector;
    vector<float> gold_a{0, 1, 2, 3, 4};
    vector<float> gold_b{0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    vector<float> gold_c{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};

    ASSERT_VEC_ARRAY_EQ(gold_a, a.dims(), a);
    ASSERT_VEC_ARRAY_EQ(gold_b, b.dims(), b);
    ASSERT_VEC_ARRAY_EQ(gold_c, c.dims(), c);
}
