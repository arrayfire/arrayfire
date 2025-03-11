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
#include <testHelpers.hpp>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>

using namespace af;
using std::vector;

template<typename T>
class Array : public ::testing::Test {};

template<typename T>
class SubArray : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, char, unsigned char,
                         int, uint, intl, uintl, short, ushort,
                         half_float::half>
    TestTypes;

typedef ::testing::Types<af_half, cdouble, cfloat, double, float, int, intl, short, uint, uintl, ushort> SubArrayTypes;

TYPED_TEST_SUITE(Array, TestTypes);
TYPED_TEST_SUITE(SubArray, SubArrayTypes);

TEST(Array, ConstructorDefault) {
    array a;
    EXPECT_EQ(0u, a.numdims());
    EXPECT_EQ(dim_t(0), a.dims(0));
    EXPECT_EQ(dim_t(0), a.elements());
    EXPECT_EQ(f32, a.type());
    EXPECT_EQ(0u, a.bytes());
    EXPECT_FALSE(a.isrow());
    EXPECT_FALSE(a.iscomplex());
    EXPECT_FALSE(a.isdouble());
    EXPECT_FALSE(a.isbool());

    EXPECT_FALSE(a.isvector());
    EXPECT_FALSE(a.iscolumn());

    EXPECT_TRUE(a.isreal());
    EXPECT_TRUE(a.isempty());
    EXPECT_TRUE(a.issingle());
    EXPECT_TRUE(a.isfloating());
    EXPECT_TRUE(a.isrealfloating());
}

TYPED_TEST(Array, ConstructorEmptyDim4) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    dim4 dims(3, 3, 3, 3);
    array a(dims, type);
    EXPECT_EQ(4u, a.numdims());
    EXPECT_EQ(dim_t(3), a.dims(0));
    EXPECT_EQ(dim_t(3), a.dims(1));
    EXPECT_EQ(dim_t(3), a.dims(2));
    EXPECT_EQ(dim_t(3), a.dims(3));
    EXPECT_EQ(dim_t(81), a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorEmpty1D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a(2, type);
    EXPECT_EQ(1u, a.numdims());
    EXPECT_EQ(dim_t(2), a.dims(0));
    EXPECT_EQ(dim_t(1), a.dims(1));
    EXPECT_EQ(dim_t(1), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(2), a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorEmpty2D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a(2, 2, type);
    EXPECT_EQ(2u, a.numdims());
    EXPECT_EQ(dim_t(2), a.dims(0));
    EXPECT_EQ(dim_t(2), a.dims(1));
    EXPECT_EQ(dim_t(1), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(4), a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorEmpty3D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a(2, 2, 2, type);
    EXPECT_EQ(3u, a.numdims());
    EXPECT_EQ(dim_t(2), a.dims(0));
    EXPECT_EQ(dim_t(2), a.dims(1));
    EXPECT_EQ(dim_t(2), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(8), a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorEmpty4D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a(2, 2, 2, 2, type);
    EXPECT_EQ(4u, a.numdims());
    EXPECT_EQ(dim_t(2), a.dims(0));
    EXPECT_EQ(dim_t(2), a.dims(1));
    EXPECT_EQ(dim_t(2), a.dims(2));
    EXPECT_EQ(dim_t(2), a.dims(3));
    EXPECT_EQ(dim_t(16), a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorHostPointer1D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type    = (dtype)dtype_traits<TypeParam>::af_type;
    size_t nelems = 10;
    vector<TypeParam> data(nelems, TypeParam(4));
    array a(nelems, &data.front(), afHost);
    EXPECT_EQ(1u, a.numdims());
    EXPECT_EQ(dim_t(nelems), a.dims(0));
    EXPECT_EQ(dim_t(1), a.dims(1));
    EXPECT_EQ(dim_t(1), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(nelems), a.elements());
    EXPECT_EQ(type, a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer2D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type      = (dtype)dtype_traits<TypeParam>::af_type;
    size_t ndims    = 2;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size;
    vector<TypeParam> data(nelems, TypeParam(4));
    array a(dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims, a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(1), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(nelems), a.elements());
    EXPECT_EQ(type, a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer3D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type      = (dtype)dtype_traits<TypeParam>::af_type;
    size_t ndims    = 3;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size * dim_size;
    vector<TypeParam> data(nelems, TypeParam(4));
    array a(dim_size, dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims, a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(dim_size), a.dims(2));
    EXPECT_EQ(dim_t(1), a.dims(3));
    EXPECT_EQ(dim_t(nelems), a.elements());
    EXPECT_EQ(type, a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer4D) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type      = (dtype)dtype_traits<TypeParam>::af_type;
    size_t ndims    = 4;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size * dim_size * dim_size;
    vector<TypeParam> data(nelems, TypeParam(4));
    array a(dim_size, dim_size, dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims, a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(dim_size), a.dims(2));
    EXPECT_EQ(dim_t(dim_size), a.dims(3));
    EXPECT_EQ(dim_t(nelems), a.elements());
    EXPECT_EQ(type, a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, TypeAttributes) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array one(10, type);
    switch (type) {
        case f32:
            EXPECT_TRUE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_TRUE(one.issingle());
            EXPECT_TRUE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;

        case f64:
            EXPECT_TRUE(one.isfloating());
            EXPECT_TRUE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_TRUE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case c32:
            EXPECT_TRUE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_TRUE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_FALSE(one.isreal());
            EXPECT_TRUE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case c64:
            EXPECT_TRUE(one.isfloating());
            EXPECT_TRUE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_FALSE(one.isreal());
            EXPECT_TRUE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case s32:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case u32:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case s16:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case u16:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case u8:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case b8:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_TRUE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case s64:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case u64:
            EXPECT_FALSE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_FALSE(one.isrealfloating());
            EXPECT_TRUE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_FALSE(one.ishalf());
            break;
        case f16:
            EXPECT_TRUE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_FALSE(one.issingle());
            EXPECT_TRUE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
            EXPECT_TRUE(one.ishalf());
            break;
    }
}

TEST(Array, ShapeAttributes) {
    dim_t dim_size = 10;
    array scalar(1);
    array col(dim_size);
    array row(1, dim_size);
    array matrix(dim_size, dim_size);
    array volume(dim_size, dim_size, dim_size);
    array hypercube(dim_size, dim_size, dim_size, dim_size);

    EXPECT_FALSE(scalar.isempty());
    EXPECT_FALSE(col.isempty());
    EXPECT_FALSE(row.isempty());
    EXPECT_FALSE(matrix.isempty());
    EXPECT_FALSE(volume.isempty());
    EXPECT_FALSE(hypercube.isempty());

    EXPECT_TRUE(scalar.isscalar());
    EXPECT_FALSE(col.isscalar());
    EXPECT_FALSE(row.isscalar());
    EXPECT_FALSE(matrix.isscalar());
    EXPECT_FALSE(volume.isscalar());
    EXPECT_FALSE(hypercube.isscalar());

    EXPECT_FALSE(scalar.isvector());
    EXPECT_TRUE(col.isvector());
    EXPECT_TRUE(row.isvector());
    EXPECT_FALSE(matrix.isvector());
    EXPECT_FALSE(volume.isvector());
    EXPECT_FALSE(hypercube.isvector());

    EXPECT_FALSE(scalar.isrow());
    EXPECT_FALSE(col.isrow());
    EXPECT_TRUE(row.isrow());
    EXPECT_FALSE(matrix.isrow());
    EXPECT_FALSE(volume.isrow());
    EXPECT_FALSE(hypercube.isrow());

    EXPECT_FALSE(scalar.iscolumn());
    EXPECT_TRUE(col.iscolumn());
    EXPECT_FALSE(row.iscolumn());
    EXPECT_FALSE(matrix.iscolumn());
    EXPECT_FALSE(volume.iscolumn());
    EXPECT_FALSE(hypercube.iscolumn());
}

TEST(Array, ISSUE_951) {
    // This works
    // const array a(100, 100);
    // array b = a.cols(0, 20);
    // b = b.rows(10, 20);

    // This works
    // array a(100, 100);
    // array b = a.cols(0, 20).rows(10, 20);

    // This fails with linking error
    const array a = randu(100, 100);
    array b       = a.cols(0, 20).rows(10, 20);
}

TEST(Array, ISSUE_3534) {
    // This works
    // array a = range(dim4(5,5));
    // a = a.rows(0,3).copy();
    // Following assignment failed silently without above copy()
    // a(0,0) = 1234;

    // Testing rows
    {
        array a = range(dim4(5, 5));
        af_array before_handle = a.get();
        const float *before_dev_pointer = a.device<float>();

        a       = a.rows(1, 4);
        af_array after_handle = a.get();
        const float *after_dev_pointer = a.device<float>();

        // We expect a copy to be created, so the handle and 
        // array pointers should be different
        ASSERT_NE(before_handle, after_handle);
        ASSERT_NE(before_dev_pointer, after_dev_pointer);

        a(1, 1) = -1234;

        array b = range(dim4(4, 5)) + 1.0;
        b(1, 1) = -1234;

        ASSERT_ARRAYS_EQ(a, b);
    }

    // Testing columns
    {
        array a                         = range(dim4(5, 5));
        af_array before_handle          = a.get();
        const float *before_dev_pointer = a.device<float>();

        a                              = a.cols(1, 4);
        af_array after_handle          = a.get();
        const float *after_dev_pointer = a.device<float>();

        // We expect a copy to be created, so the handle and
        // array pointers should be different
        ASSERT_NE(before_handle, after_handle);
        ASSERT_NE(before_dev_pointer, after_dev_pointer);

        a(1, 1) = -1234;

        array b = range(dim4(5, 4));
        b(1, 1) = -1234;

        ASSERT_ARRAYS_EQ(a, b);
    }

    // Testing subarrays with sizes of one page
    {
        array a     = range(dim4(64, 64));
        a           = a.rows(31, 57);
        a(1, 1)     = -123456;

        array b     = range(dim4(27, 64)) + 31.0;
        b(1, 1)     = -123456;

        ASSERT_ARRAYS_EQ(a, b);

        a           = range(dim4(128, 128));
        a           = a.rows(0, 63);
        a           = a.cols(0, 63);
        a(0, 0)     = -54321;
        a(63, 63)   = -12345;

        b = range(dim4(64, 64));
        b(0, 0)     = -54321;
        b(63, 63)   = -12345;

        ASSERT_ARRAYS_EQ(a, b);

        a             = range(dim4(128, 128));
        a(64, 64)   = -67890;
        a(127, 127)   = -9876;
        a             = a.rows(64, 127);
        a             = a.cols(64, 127);
        
        a(0, 0)   = -54321;
        a(63, 63) = -12345;

        b         = range(dim4(64, 64)) + 64;
        b(0, 0)   = -54321;
        b(63, 63) = -12345;

        ASSERT_ARRAYS_EQ(a, b);
    }
}

TEST(Array, CreateHandleInvalidNullDimsPointer) {
    af_array out = 0;
    EXPECT_EQ(AF_ERR_ARG, af_create_handle(&out, 1, NULL, f32));
}

template<typename T, typename Id>
void subArrayIndex() {
    array A1a = randu(100, (af_dtype)dtype_traits<T>::af_type);
    array B1a = A1a(seq(3, 75)).copy();
    A1a = A1a(seq(3, 75));
    A1a(seq(10,20)) = 9999;
    B1a(seq(10,20)) = 9999;
    ASSERT_ARRAYS_EQ(A1a, B1a);

    array A1b = randu(100, (af_dtype)dtype_traits<T>::af_type);
    array B1b = A1b(seq(3, 75)).copy();
    A1b = A1b(seq(3, 75));
    array idx1b = randu(B1b.dims()[0] / 4, u32) % B1b.dims()[0];
          idx1b = idx1b.as((af_dtype)dtype_traits<Id>::af_type);
    A1b(idx1b) = 9999;
    B1b(idx1b) = 9999;
    ASSERT_ARRAYS_EQ(A1b, B1b);

    array A1c = randu(100, (af_dtype)dtype_traits<T>::af_type);
    array idx1c = randu(50, u32) % 100;
          idx1c = idx1c.as((af_dtype)dtype_traits<Id>::af_type);
    array B1c = A1c(idx1c).copy();
    A1c = A1c(idx1c);
    idx1c = randu(B1c.dims()[0] / 4, u32) % B1c.dims()[0];
    idx1c = idx1c.as((af_dtype)dtype_traits<Id>::af_type);
    A1c(idx1c) = 9999;
    B1c(idx1c) = 9999;
    ASSERT_ARRAYS_EQ(A1c, B1c);

    array A2a = randu(10, 19, (af_dtype)dtype_traits<T>::af_type);
    array B2a = A2a.rows(2,7).copy();
    A2a = A2a.rows(2,7);
    array idx2a = randu(5, u32) % 19;
          idx2a = idx2a.as((af_dtype)dtype_traits<Id>::af_type);
    A2a(5, idx2a) = 9999;
    B2a(5, idx2a) = 9999;
    A2a(seq(2,4), 1) = 9999;
    B2a(seq(2,4), 1) = 9999;
    A2a(1, 2) = 9999;
    B2a(1, 2) = 9999;
    ASSERT_ARRAYS_EQ(A2a, B2a);

    array A2b = randu(10, 19, (af_dtype)dtype_traits<T>::af_type);
    array B2b = A2b.cols(10,17).copy();
    A2b = A2b.cols(10,17);
    array idx2b = randu(5, u32) % 19;
          idx2b = idx2b.as((af_dtype)dtype_traits<Id>::af_type);
    A2b(5, idx2b) = 9999;
    B2b(5, idx2b) = 9999;
    A2b(seq(2,4), 1) = 9999;
    B2b(seq(2,4), 1) = 9999;
    A2b(1, 2) = 9999;
    B2b(1, 2) = 9999;
    ASSERT_ARRAYS_EQ(A2b, B2b);

    array A2c = randu(10, 19, (af_dtype)dtype_traits<T>::af_type);
    array B2c = A2c(seq(2,7), span).copy();
    A2c = A2c(seq(2,7), span);
    array idx2c = randu(5, u32) % 19;
          idx2c = idx2c.as((af_dtype)dtype_traits<Id>::af_type);
    A2c(5, idx2c) = 9999;
    B2c(5, idx2c) = 9999;
    A2c(seq(2,4), 1) = 9999;
    B2c(seq(2,4), 1) = 9999;
    A2c(1, 2) = 9999;
    B2c(1, 2) = 9999;
    ASSERT_ARRAYS_EQ(A2c, B2c);

    array A2d = randu(10, 19, (af_dtype)dtype_traits<T>::af_type);
    array B2d = A2d(span, seq(10,16)).copy();
    A2d = A2d(span, seq(10,16));
    array idx2d = randu(3, u32) % 6;
          idx2d = idx2d.as((af_dtype)dtype_traits<Id>::af_type);
    A2d(5, idx2d) = 9999;
    B2d(5, idx2d) = 9999;
    A2d(seq(2,4), 1) = 9999;
    B2d(seq(2,4), 1) = 9999;
    A2d(1, 2) = 9999;
    B2d(1, 2) = 9999;
    ASSERT_ARRAYS_EQ(A2d, B2d);

    array A4a = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array B4a = A4a(seq(3,9), span, span, span).copy();
    A4a = A4a(seq(3,9), span, span, span);
    array idx4a = randu(10, u32) % 19;
          idx4a = idx4a.as((af_dtype)dtype_traits<Id>::af_type);
    A4a(5, idx4a, 4, 1) = 9999;
    B4a(5, idx4a, 4, 1) = 9999;
    A4a(seq(2,4), 1, 4, 1) = 9999;
    B4a(seq(2,4), 1, 4, 1) = 9999;
    A4a(1, 2, 4, 1) = 9999;
    B4a(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4a, B4a);

    array A4b = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array B4b = A4b(span, seq(2,15), span, span).copy();
    A4b = A4b(span, seq(2,15), span, span);
    array idx4b = randu(5, u32) % 10;
          idx4b = idx4b.as((af_dtype)dtype_traits<Id>::af_type);
    A4b(5, idx4b, 4, 1) = 9999;
    B4b(5, idx4b, 4, 1) = 9999;
    A4b(seq(2,4), 1, 4, 1) = 9999;
    B4b(seq(2,4), 1, 4, 1) = 9999;
    A4b(1, 2, 4, 1) = 9999;
    B4b(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4b, B4b);

    array A4c = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array B4c = A4c(span, span, span, seq(1,2)).copy();
    A4c = A4c(span, span, span, seq(1,2));
    array idx4c = randu(5, u32) % 10;
          idx4c = idx4c.as((af_dtype)dtype_traits<Id>::af_type);
    A4c(5, idx4c, 4, 1) = 9999;
    B4c(5, idx4c, 4, 1) = 9999;
    A4c(seq(2,4), 1, 4, 1) = 9999;
    B4c(seq(2,4), 1, 4, 1) = 9999;
    A4c(1, 2, 4, 1) = 9999;
    B4c(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4c, B4c);

    array A4d = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4d1 = randu(10, u32) % 19;
          idx4d1 = idx4d1.as((af_dtype)dtype_traits<Id>::af_type);
    array B4d = A4d(seq(3,9), idx4d1, span, span).copy();
    A4d = A4d(seq(3,9), idx4d1, span, span);
    array idx4d2 = randu(5, u32) % 10;
          idx4d2 = idx4d2.as((af_dtype)dtype_traits<Id>::af_type);
    A4d(5, idx4d2, 4, 1) = 9999;
    B4d(5, idx4d2, 4, 1) = 9999;
    A4d(seq(2,4), 1, 4, 1) = 9999;
    B4d(seq(2,4), 1, 4, 1) = 9999;
    A4d(1, 2, 4, 1) = 9999;
    B4d(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4d, B4d);

    array A4e = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4e1 = randu(6, u32) % 10;
          idx4e1 = idx4e1.as((af_dtype)dtype_traits<Id>::af_type);
    array B4e = A4e(idx4e1, span, span, span).copy();
    A4e = A4e(idx4e1, span, span, span);
    array idx4e2 = randu(5, u32) % 10;
          idx4e2 = idx4e2.as((af_dtype)dtype_traits<Id>::af_type);
    A4e(1, idx4e2, 4, 1) = 9999;
    B4e(1, idx4e2, 4, 1) = 9999;
    A4e(seq(1,2), 1, 4, 1) = 9999;
    B4e(seq(1,2), 1, 4, 1) = 9999;
    A4e(1, 2, 4, 1) = 9999;
    B4e(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4e, B4e);

    array A4f = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4f1 = randu(10, u32) % 19;
          idx4f1 = idx4f1.as((af_dtype)dtype_traits<Id>::af_type);
    array B4f = A4f(span, idx4f1, span, span).copy();
    A4f = A4f(span, idx4f1, span, span);
    array idx4f2 = randu(5, u32) % 10;
          idx4f2 = idx4f2.as((af_dtype)dtype_traits<Id>::af_type);
    A4f(1, idx4f2, 4, 1) = 9999;
    B4f(1, idx4f2, 4, 1) = 9999;
    A4f(seq(1,2), 1, 4, 1) = 9999;
    B4f(seq(1,2), 1, 4, 1) = 9999;
    A4f(1, 2, 4, 1) = 9999;
    B4f(1, 2, 4, 1) = 9999;
    ASSERT_ARRAYS_EQ(A4f, B4f);
}

template<typename T>
void subArrayIndex() {
    int dev = af::getDevice();
    if(af::isHalfAvailable(dev)) subArrayIndex<T, af_half>();
    if(af::isDoubleAvailable(dev)) subArrayIndex<T, double>();
    subArrayIndex<T, float>();
    subArrayIndex<T, int>();
    subArrayIndex<T, intl>();
    subArrayIndex<T, short>();
    subArrayIndex<T, uint>();
    subArrayIndex<T, uintl>();
    subArrayIndex<T, ushort>();
}

TYPED_TEST(SubArray, SubArrayOfSubArrayWorks_ISSUE_3534) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    //Assigning to a subarray of a subarray should work
    subArrayIndex<TypeParam>();
}
TEST(Device, simple) {
    array a = randu(5, 5);
    {
        float *ptr0 = a.device<float>();
        float *ptr1 = a.device<float>();
        ASSERT_EQ(ptr0, ptr1);
    }

    {
        float *ptr0 = a.device<float>();
        a.unlock();
        float *ptr1 = a.device<float>();
        ASSERT_EQ(ptr0, ptr1);
    }
}

TEST(Device, index) {
    array a = randu(5, 5);
    array b = a(span, 0);

    ASSERT_NE(a.device<float>(), b.device<float>());
}

TEST(Device, unequal) {
    {
        array a    = randu(5, 5);
        float *ptr = a.device<float>();
        array b    = a;
        ASSERT_NE(ptr, b.device<float>());
        ASSERT_EQ(ptr, a.device<float>());
    }

    {
        array a    = randu(5, 5);
        float *ptr = a.device<float>();
        array b    = a;
        ASSERT_NE(ptr, a.device<float>());
        ASSERT_EQ(ptr, b.device<float>());
    }
}

TEST(DeviceId, Same) {
    array a = randu(5, 5);
    ASSERT_EQ(getDevice(), getDeviceId(a));
}

TEST(DeviceId, Different) {
    int ndevices = getDeviceCount();
    if (ndevices < 2) GTEST_SKIP() << "Skipping mult-GPU test";
    int id0 = getDevice();
    int id1 = (id0 + 1) % ndevices;

    {
        array a = randu(5, 5);
        ASSERT_EQ(getDeviceId(a), id0);
        setDevice(id1);

        array b = randu(5, 5);

        ASSERT_EQ(getDeviceId(a), id0);
        ASSERT_EQ(getDeviceId(b), id1);
        ASSERT_NE(getDevice(), getDeviceId(a));
        ASSERT_EQ(getDevice(), getDeviceId(b));

        af_array c;
        af_err err = af_matmul(&c, a.get(), b.get(), AF_MAT_NONE, AF_MAT_NONE);
        af::sync();
        ASSERT_EQ(err, AF_SUCCESS);
    }

    setDevice(id1);
    deviceGC();
    setDevice(id0);
    deviceGC();
}

TEST(Device, MigrateAllDevicesToAllDevices) {
    int ndevices = getDeviceCount();
    if (ndevices < 2) GTEST_SKIP() << "Skipping mult-GPU test";

    for (int i = 0; i < ndevices; i++) {
        for (int j = 0; j < ndevices; j++) {
            setDevice(i);
            array a = constant(i * 255, 10, 10);
            a.eval();

            setDevice(j);
            array b = constant(j * 256, 10, 10);
            b.eval();

            array c = a + b;

            std::vector<float> gold(10 * 10, i * 255 + j * 256);

            ASSERT_VEC_ARRAY_EQ(gold, dim4(10, 10), c);
        }
    }
}

TEST(Device, empty) {
    array a = array();
    ASSERT_EQ(a.device<float>(), nullptr);
}

TEST(Device, JIT) {
    array a = constant(1, 5, 5);
    ASSERT_NE(a.device<float>(), nullptr);
}

TYPED_TEST(Array, Scalar) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a    = randu(dim4(1), type);

    vector<TypeParam> gold(a.elements());

    a.host((void *)gold.data());

    EXPECT_EQ(gold[0], a.scalar<TypeParam>());
}

TEST(Array, ScalarTypeMismatch) {
    array a = constant(1.0, dim4(1), f32);

    EXPECT_THROW(a.scalar<int>(), exception);
}

TEST(Array, CopyListInitializerList) {
    int h_buffer[] = {23, 34, 18, 99, 34};

    array A(5, h_buffer);
    array B({23, 34, 18, 99, 34});

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, DirectListInitializerList2) {
    int h_buffer[] = {23, 34, 18, 99, 34};

    array A(5, h_buffer);
    array B{23, 34, 18, 99, 34};

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, CopyListInitializerListAndDim4) {
    int h_buffer[] = {23, 34, 18, 99, 34, 44};

    array A(2, 3, h_buffer);
    array B(dim4(2, 3), {23, 34, 18, 99, 34, 44});

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, DirectListInitializerListAndDim4) {
    int h_buffer[] = {23, 34, 18, 99, 34, 44};

    array A(2, 3, h_buffer);
    array B{dim4(2, 3), {23, 34, 18, 99, 34, 44}};

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, CopyListInitializerListAssignment) {
    int h_buffer[] = {23, 34, 18, 99, 34};

    array A(5, h_buffer);
    array B = {23, 34, 18, 99, 34};

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, CopyListInitializerListDim4Assignment) {
    int h_buffer[] = {23, 34, 18, 99, 34, 44};

    array A(2, 3, h_buffer);
    array B = {dim4(2, 3), {23, 34, 18, 99, 34, 44}};

    ASSERT_ARRAYS_EQ(A, B);
}

TEST(Array, EmptyArrayHostCopy) {
    af::array empty;
    std::vector<float> hdata(100);
    empty.host(hdata.data());
    SUCCEED();
}

TEST(Array, ReferenceCount1) {
    int counta = 0, countb = 0, countc = 0;
    array a = af::randu(10, 10);
    a.eval();
    af::sync();
    {
        ASSERT_REF(a, 1) << "After a = randu(10, 10);";

        array b = af::randu(10, 10);  //(af::seq(100));
        ASSERT_REF(b, 1) << "After b = randu(10, 10);";

        array c = a + b;
        ASSERT_REF(a, 2) << "After c = a + b;";
        ASSERT_REF(b, 2) << "After c = a + b;";
        ASSERT_REF(c, 0) << "After c = a + b;";

        c.eval();
        af::sync();
        ASSERT_REF(a, 1) << "After c.eval();";
        ASSERT_REF(b, 1) << "After c.eval();";
        ASSERT_REF(c, 1) << "After c.eval();";
    }
}

TEST(Array, ReferenceCount2) {
    int counta = 0, countb = 0, countc = 0;
    array a = af::randu(10, 10);
    array b = af::randu(10, 10);
    {
        ASSERT_REF(a, 1) << "After a = randu(10, 10);";
        ASSERT_REF(b, 1) << "After a = randu(10, 10);";

        array c = a + b;

        ASSERT_REF(a, 2) << "After c = a + b;";
        ASSERT_REF(b, 2) << "After c = a + b;";
        ASSERT_REF(c, 0) << "After c = a + b;";

        array d = c;

        ASSERT_REF(a, 2) << "After d = c;";
        ASSERT_REF(b, 2) << "After d = c;";
        ASSERT_REF(c, 0) << "After d = c;";
        ASSERT_REF(d, 0) << "After d = c;";
    }
}

// This tests situations where the compiler incorrectly assumes the
// initializer list constructor instead of the regular constructor when
// using the uniform initilization syntax
TEST(Array, InitializerListFixAFArray) {
    af::array a = randu(1);
    af::array b{a};

    ASSERT_ARRAYS_EQ(a, b);
}

// This tests situations where the compiler incorrectly assumes the
// initializer list constructor instead of the regular constructor when
// using the uniform initilization syntax
TEST(Array, InitializerListFixDim4) {
    af::array a        = randu(1);
    vector<float> data = {3.14f, 3.14f, 3.14f, 3.14f, 3.14f,
                          3.14f, 3.14f, 3.14f, 3.14f};
    af::array b{dim4(3, 3), data.data()};
    ASSERT_ARRAYS_EQ(constant(3.14, 3, 3), b);
}

TEST(Array, OtherDevice) {
    if (af::getDeviceCount() == 1) GTEST_SKIP() << "Single device. Skipping";
    af::setDevice(0);
    af::info();
    af::array a = constant(3, 5, 5);
    a.eval();
    af::setDevice(1);
    af::info();
    af::array b = constant(2, 5, 5);
    b.eval();

    af::array c = a + b;
    af::eval(c);
    af::sync();
    af::setDevice(0);
    ASSERT_ARRAYS_EQ(constant(5, 5, 5), c);
}
