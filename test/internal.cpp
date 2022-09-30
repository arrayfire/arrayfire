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
#include <af/dim4.hpp>
#include <af/internal.h>
#include <af/traits.hpp>
#include <string>
#include <vector>

using af::array;
using af::dim4;
using af::randu;
using af::seq;
using af::span;
using std::vector;

TEST(Internal, CreateStrided) {
    float ha[] = {1,    101,  102,  103,  104,  105,  201,  202,  203,  204,
                  205,  301,  302,  303,  304,  305,  401,  402,  403,  404,
                  405,

                  1010, 1020, 1030, 1040, 1050, 2010, 2020, 2030, 2040, 2050,
                  3010, 3020, 3030, 3040, 3050, 4010, 4020, 4030, 4040, 4050};

    dim_t offset    = 1;
    unsigned ndims  = 3;
    dim_t dims[]    = {3, 3, 2};
    dim_t strides[] = {1, 5, 20};
    array a         = createStridedArray((void *)ha, offset, dim4(ndims, dims),
                                         dim4(ndims, strides), f32, afHost);

    dim4 astrides = getStrides(a);
    dim4 adims    = a.dims();

    ASSERT_EQ(offset, getOffset(a));
    for (int i = 0; i < (int)ndims; i++) {
        ASSERT_EQ(strides[i], astrides[i]);
        ASSERT_EQ(dims[i], adims[i]);
    }

    vector<float> va(a.elements());
    a.host(&va[0]);

    int o = offset;
    for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int i = 0; i < dims[0]; i++) {
                ASSERT_EQ(
                    va[i + j * dims[0] + k * dims[0] * dims[1]],
                    ha[i * strides[0] + j * strides[1] + k * strides[2] + o])
                    << "at (" << i << "," << j << "," << k << ")";
            }
        }
    }
}

TEST(Internal, CheckInfo) {
    const int xdim = 10;
    const int ydim = 8;

    const int xoff = 1;
    const int yoff = 2;

    const int xnum = 5;
    const int ynum = 3;

    array a = randu(10, 8);

    array b = a(seq(xoff, xoff + xnum - 1), seq(yoff, yoff + ynum - 1));

    dim4 strides = getStrides(b);
    dim4 dims    = b.dims();

    dim_t offset = xoff + yoff * xdim;

    ASSERT_EQ(dims[0], xnum);
    ASSERT_EQ(dims[1], ynum);
    ASSERT_EQ(isOwner(a), true);
    ASSERT_EQ(isOwner(b), false);

    ASSERT_EQ(getOffset(b), offset);
    ASSERT_EQ(strides[0], 1);
    ASSERT_EQ(strides[1], xdim);
    ASSERT_EQ(strides[2], xdim * ydim);
    ASSERT_EQ(getRawPtr(a), getRawPtr(b));
}

TEST(Internal, Linear) {
    array c;
    {
        array a = randu(10, 8);

        // b is just pointing to same underlying data
        // b is an owner;
        array b = a;
        ASSERT_EQ(isOwner(b), true);

        // C is considered sub array
        // C will not be an owner
        c = a(span);
        ASSERT_EQ(isOwner(c), false);
    }

    // Even though a and b are out of scope, c is still not an owner
    { ASSERT_EQ(isOwner(c), false); }
}

TEST(Internal, Allocated) {
    array a            = randu(10, 8);
    size_t a_allocated = a.allocated();
    size_t a_bytes     = a.bytes();

    // b is just pointing to same underlying data
    // b is an owner;
    array b = a;
    ASSERT_EQ(b.allocated(), a_allocated);
    ASSERT_EQ(b.bytes(), a_bytes);

    // C is considered sub array
    // C will not be an owner
    array c = a(span);
    ASSERT_EQ(c.allocated(), a_allocated);
    ASSERT_EQ(c.bytes(), a_bytes);

    array d = a.col(1);
    ASSERT_EQ(d.allocated(), a_allocated);

    a = randu(20);
    b = randu(20);

    // Even though a, b are reallocated and c, d are not owners
    // the allocated and bytes should remain the same
    ASSERT_EQ(c.allocated(), a_allocated);
    ASSERT_EQ(c.bytes(), a_bytes);

    ASSERT_EQ(d.allocated(), a_allocated);
}
