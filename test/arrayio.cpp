/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>

#include <testHelpers.hpp>

#include <complex>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::constant;
using af::dim4;
using af::readArray;
using af::saveArray;
using std::complex;
using std::string;
using std::vector;

struct type_params {
    string name;
    af_dtype type;
    double real;
    double imag;
    type_params(string n, af_dtype t, double r, double i = 0.)
        : name(n), type(t), real(r), imag(i) {}
};

class ArrayIOType : public ::testing::TestWithParam<type_params> {};

string getTypeName(
    const ::testing::TestParamInfo<ArrayIOType::ParamType> info) {
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    Types, ArrayIOType,
    ::testing::Values(type_params("f32", f32, 3.14f, 0),
                      type_params("f64", f64, 3.14, 0),
                      type_params("c32", c32, 3.0f, 4.5f),
                      type_params("c64", c64, 3.0, 4.5),
                      type_params("s32", s32, 11), type_params("u32", u32, 12),
                      type_params("u8", u8, 13), type_params("b8", b8, 1),
                      type_params("s64", s64, 15), type_params("u64", u64, 16),
                      type_params("s16", s16, 17), type_params("u16", u16, 18)),
    getTypeName);

TEST_P(ArrayIOType, ReadType) {
    type_params p = GetParam();
    if (noDoubleTests(p.type)) GTEST_SKIP() << "No double support.";
    array arr =
        readArray((string(TEST_DIR) + "/arrayio/" + p.name + ".arr").c_str(),
                  p.name.c_str());

    ASSERT_EQ(arr.type(), p.type);
}

TEST_P(ArrayIOType, ReadSize) {
    type_params p = GetParam();
    if (noDoubleTests(p.type)) GTEST_SKIP() << "No double support.";
    array arr =
        readArray((string(TEST_DIR) + "/arrayio/" + p.name + ".arr").c_str(),
                  p.name.c_str());

    ASSERT_EQ(arr.dims(), dim4(10, 10));
}

template<typename T>
void checkVals(array arr, double r, double i, af_dtype t) {
    vector<T> d(arr.elements());
    arr.host(d.data());
    int elements = arr.elements();
    for (int ii = 0; ii < elements; ii++) {
        if (t == c32 || t == c64) {
            ASSERT_EQ(r, real<T>(d[ii])) << "at: " << ii;
            ASSERT_EQ(i, imag<T>(d[ii])) << "at: " << ii;
        } else {
            ASSERT_EQ(real(r), real(d[ii])) << "at: " << ii;
        }
    }
}

TEST_P(ArrayIOType, ReadContent) {
    type_params p = GetParam();
    if (noDoubleTests(p.type)) GTEST_SKIP() << "No double support.";
    array arr =
        readArray((string(TEST_DIR) + "/arrayio/" + p.name + ".arr").c_str(),
                  p.name.c_str());

    switch (arr.type()) {
        case f32: checkVals<float>(arr, p.real, p.imag, p.type); break;
        case f64: checkVals<double>(arr, p.real, p.imag, p.type); break;
        case c32: checkVals<af::cfloat>(arr, p.real, p.imag, p.type); break;
        case c64: checkVals<af::cdouble>(arr, p.real, p.imag, p.type); break;
        case s32: checkVals<int>(arr, p.real, p.imag, p.type); break;
        case u32: checkVals<unsigned>(arr, p.real, p.imag, p.type); break;
        case u8: checkVals<unsigned char>(arr, p.real, p.imag, p.type); break;
        case b8: checkVals<char>(arr, p.real, p.imag, p.type); break;
        case s64: checkVals<long long>(arr, p.real, p.imag, p.type); break;
        case u64:
            checkVals<unsigned long long>(arr, p.real, p.imag, p.type);
            break;
        case s16: checkVals<short>(arr, p.real, p.imag, p.type); break;
        case u16: checkVals<unsigned short>(arr, p.real, p.imag, p.type); break;
        default: FAIL() << "Invalid type";
    }
}

TEST(ArrayIO, Save) {
    array a = constant(1, 10, 10);
    array b = constant(2, 10, 10);

    saveArray("a", a, "arr.af");
    saveArray("b", b, "arr.af", true);

    array aread = readArray("arr.af", "a");
    array bread = readArray("arr.af", "b");

    ASSERT_ARRAYS_EQ(a, aread);
    ASSERT_ARRAYS_EQ(b, bread);
}
