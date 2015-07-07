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
#include <testHelpers.hpp>

using namespace af;
using std::vector;

template<typename T>
class Array : public ::testing::Test
{

};

typedef ::testing::Types<float, double, af::cfloat, af::cdouble, char, unsigned char, int, uint, intl, uintl> TestTypes;
TYPED_TEST_CASE(Array, TestTypes);

TEST(Array, ConstructorDefault)
{
    array a;
    EXPECT_EQ(0u,    a.numdims());
    EXPECT_EQ(dim_t(0),    a.dims(0));
    EXPECT_EQ(dim_t(0),    a.dims(1));
    EXPECT_EQ(dim_t(0),    a.dims(2));
    EXPECT_EQ(dim_t(0),    a.dims(3));
    EXPECT_EQ(dim_t(0),    a.elements());
    EXPECT_EQ(f32,  a.type());
    EXPECT_EQ(0u,    a.bytes());
    EXPECT_FALSE(   a.isrow());
    EXPECT_FALSE(   a.iscomplex());
    EXPECT_FALSE(   a.isdouble());
    EXPECT_FALSE(   a.isbool());

    EXPECT_FALSE(    a.isvector());
    EXPECT_FALSE(    a.iscolumn());

    EXPECT_TRUE(    a.isreal());
    EXPECT_TRUE(    a.isempty());
    EXPECT_TRUE(    a.issingle());
    EXPECT_TRUE(    a.isfloating());
    EXPECT_TRUE(    a.isrealfloating());
}

TYPED_TEST(Array, ConstructorEmptyDim4)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    dim4 dims(3, 3, 3, 3);
    array a(dims, type);
    EXPECT_EQ(4u,    a.numdims());
    EXPECT_EQ(dim_t(3),    a.dims(0));
    EXPECT_EQ(dim_t(3),    a.dims(1));
    EXPECT_EQ(dim_t(3),    a.dims(2));
    EXPECT_EQ(dim_t(3),    a.dims(3));
    EXPECT_EQ(dim_t(81),   a.elements());
    EXPECT_EQ(type,  a.type());
}

TYPED_TEST(Array, ConstructorEmpty1D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    array a(2, type);
    EXPECT_EQ(1u,    a.numdims());
    EXPECT_EQ(dim_t(2),    a.dims(0));
    EXPECT_EQ(dim_t(1),    a.dims(1));
    EXPECT_EQ(dim_t(1),    a.dims(2));
    EXPECT_EQ(dim_t(1),    a.dims(3));
    EXPECT_EQ(dim_t(2),    a.elements());
    EXPECT_EQ(type,  a.type());
}

TYPED_TEST(Array, ConstructorEmpty2D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    array a(2, 2, type);
    EXPECT_EQ(2u,    a.numdims());
    EXPECT_EQ(dim_t(2),    a.dims(0));
    EXPECT_EQ(dim_t(2),    a.dims(1));
    EXPECT_EQ(dim_t(1),    a.dims(2));
    EXPECT_EQ(dim_t(1),    a.dims(3));
    EXPECT_EQ(dim_t(4),    a.elements());
    EXPECT_EQ(type,  a.type());
}

TYPED_TEST(Array, ConstructorEmpty3D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    array a(2, 2, 2, type);
    EXPECT_EQ(3u,    a.numdims());
    EXPECT_EQ(dim_t(2),    a.dims(0));
    EXPECT_EQ(dim_t(2),    a.dims(1));
    EXPECT_EQ(dim_t(2),    a.dims(2));
    EXPECT_EQ(dim_t(1),    a.dims(3));
    EXPECT_EQ(dim_t(8),    a.elements());
    EXPECT_EQ(type,  a.type());
}

TYPED_TEST(Array, ConstructorEmpty4D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    array a(2, 2, 2, 2, type);
    EXPECT_EQ(4u,    a.numdims());
    EXPECT_EQ(dim_t(2),    a.dims(0));
    EXPECT_EQ(dim_t(2),    a.dims(1));
    EXPECT_EQ(dim_t(2),    a.dims(2));
    EXPECT_EQ(dim_t(2),    a.dims(3));
    EXPECT_EQ(dim_t(16),   a.elements());
    EXPECT_EQ(type, a.type());
}

TYPED_TEST(Array, ConstructorHostPointer1D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    size_t nelems = 10;
    vector<TypeParam> data(nelems, 4);
    array a(nelems, &data.front(), afHost);
    EXPECT_EQ(1u,        a.numdims());
    EXPECT_EQ(dim_t(nelems),   a.dims(0));
    EXPECT_EQ(dim_t(1),        a.dims(1));
    EXPECT_EQ(dim_t(1),        a.dims(2));
    EXPECT_EQ(dim_t(1),        a.dims(3));
    EXPECT_EQ(dim_t(nelems),   a.elements());
    EXPECT_EQ(type,     a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer2D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    size_t ndims    = 2;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size;
    vector<TypeParam> data(nelems, 4);
    array a(dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims,    a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(1),        a.dims(2));
    EXPECT_EQ(dim_t(1),        a.dims(3));
    EXPECT_EQ(dim_t(nelems),   a.elements());
    EXPECT_EQ(type,     a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer3D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    size_t ndims    = 3;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size * dim_size;
    vector<TypeParam> data(nelems, 4);
    array a(dim_size, dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims,    a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(dim_size), a.dims(2));
    EXPECT_EQ(dim_t(1),        a.dims(3));
    EXPECT_EQ(dim_t(nelems),   a.elements());
    EXPECT_EQ(type,     a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, ConstructorHostPointer4D)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    size_t ndims    = 4;
    size_t dim_size = 10;
    size_t nelems   = dim_size * dim_size * dim_size * dim_size;
    vector<TypeParam> data(nelems, 4);
    array a(dim_size, dim_size, dim_size, dim_size, &data.front(), afHost);
    EXPECT_EQ(ndims,    a.numdims());
    EXPECT_EQ(dim_t(dim_size), a.dims(0));
    EXPECT_EQ(dim_t(dim_size), a.dims(1));
    EXPECT_EQ(dim_t(dim_size), a.dims(2));
    EXPECT_EQ(dim_t(dim_size), a.dims(3));
    EXPECT_EQ(dim_t(nelems),   a.elements());
    EXPECT_EQ(type,     a.type());

    vector<TypeParam> out(nelems);
    a.host(&out.front());
    ASSERT_TRUE(std::equal(data.begin(), data.end(), out.begin()));
}

TYPED_TEST(Array, TypeAttributes)
{
    if (noDoubleTests<TypeParam>()) return;

    dtype type = (dtype)af::dtype_traits<TypeParam>::af_type;
    array one(10, type);
    switch(type) {
        case f32:
            EXPECT_TRUE(one.isfloating());
            EXPECT_FALSE(one.isdouble());
            EXPECT_TRUE(one.issingle());
            EXPECT_TRUE(one.isrealfloating());
            EXPECT_FALSE(one.isinteger());
            EXPECT_TRUE(one.isreal());
            EXPECT_FALSE(one.iscomplex());
            EXPECT_FALSE(one.isbool());
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
            break;

    }

}

TEST(Array, ShapeAttributes)
{
    dim_t dim_size = 10;
    array scalar(1);
    array col(dim_size);
    array row(1, dim_size);
    array matrix(dim_size, dim_size);
    array volume(dim_size, dim_size, dim_size);
    array hypercube(dim_size, dim_size, dim_size, dim_size);

    EXPECT_FALSE(scalar.    isempty());
    EXPECT_FALSE(col.       isempty());
    EXPECT_FALSE(row.       isempty());
    EXPECT_FALSE(matrix.    isempty());
    EXPECT_FALSE(volume.    isempty());
    EXPECT_FALSE(hypercube. isempty());

    EXPECT_TRUE(scalar.     isscalar());
    EXPECT_FALSE(col.       isscalar());
    EXPECT_FALSE(row.       isscalar());
    EXPECT_FALSE(matrix.    isscalar());
    EXPECT_FALSE(volume.    isscalar());
    EXPECT_FALSE(hypercube. isscalar());

    EXPECT_FALSE(scalar.    isvector());
    EXPECT_TRUE(col.        isvector());
    EXPECT_TRUE(row.        isvector());
    EXPECT_FALSE(matrix.    isvector());
    EXPECT_FALSE(volume.    isvector());
    EXPECT_FALSE(hypercube. isvector());

    EXPECT_FALSE(scalar.    isrow());
    EXPECT_FALSE(col.       isrow());
    EXPECT_TRUE(row.        isrow());
    EXPECT_FALSE(matrix.    isrow());
    EXPECT_FALSE(volume.    isrow());
    EXPECT_FALSE(hypercube. isrow());

    EXPECT_FALSE(scalar.    iscolumn());
    EXPECT_TRUE(col.        iscolumn());
    EXPECT_FALSE(row.       iscolumn());
    EXPECT_FALSE(matrix.    iscolumn());
    EXPECT_FALSE(volume.    iscolumn());
    EXPECT_FALSE(hypercube. iscolumn());
}
