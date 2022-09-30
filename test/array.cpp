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

typedef ::testing::Types<float, double, cfloat, cdouble, char, unsigned char,
                         int, uint, intl, uintl, short, ushort,
                         half_float::half>
    TestTypes;

TYPED_TEST_SUITE(Array, TestTypes);

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

TEST(Array, CreateHandleInvalidNullDimsPointer) {
    af_array out = 0;
    EXPECT_EQ(AF_ERR_ARG, af_create_handle(&out, 1, NULL, f32));
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
    if (ndevices < 2) return;
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
        ASSERT_EQ(err, AF_ERR_DEVICE);
    }

    setDevice(id1);
    deviceGC();
    setDevice(id0);
    deviceGC();
}

TEST(Device, empty) {
    array a = array();
    ASSERT_EQ(a.device<float>() == NULL, 1);
}

TEST(Device, JIT) {
    array a = constant(1, 5, 5);
    ASSERT_EQ(a.device<float>() != NULL, 1);
}

TYPED_TEST(Array, Scalar) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dtype type = (dtype)dtype_traits<TypeParam>::af_type;
    array a    = randu(dim4(1), type);

    vector<TypeParam> gold(a.elements());

    a.host((void *)gold.data());

    EXPECT_EQ(true, gold[0] == a.scalar<TypeParam>());
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

// This tests situations where the compiler incorrectly assumes the initializer
// list constructor instead of the regular constructor when using the uniform
// initilization syntax
TEST(Array, InitializerListFixAFArray) {
    array a = randu(1);
    array b{a};

    ASSERT_ARRAYS_EQ(a, b);
}

// This tests situations where the compiler incorrectly assumes the initializer
// list constructor instead of the regular constructor when using the uniform
// initilization syntax
TEST(Array, InitializerListFixDim4) {
    array a            = randu(1);
    vector<float> data = {3.14f, 3.14f, 3.14f, 3.14f, 3.14f,
                          3.14f, 3.14f, 3.14f, 3.14f};
    array b{dim4(3, 3), data.data()};
    ASSERT_ARRAYS_EQ(constant(3.14, 3, 3), b);
}
