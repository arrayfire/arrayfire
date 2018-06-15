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
using namespace af;

TEST(GFOR, Assign_Scalar_Span)
{
    const int num = 1000;
    const float val = 3;
    array A = randu(num);

    gfor(seq ii, num) {
        A(ii) = val;
    }

    float *hA = A.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hA[i], val);
    }

    freeHost(hA);
}

TEST(GFOR, Assign_Scalar_Seq)
{
    const int num = 1000;
    const int st = 100;
    const int en = 500;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, st, en) {
        A(ii) = val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) ASSERT_EQ(hA[i], val);
        else ASSERT_EQ(hA[i], hB[i]);
    }

    freeHost(hA);
    freeHost(hB);
}

TEST(GFOR, Inc_Scalar_Span)
{
    const int num = 1000;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, num) {
        A(ii) += val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hA[i], val + hB[i]);
    }

    freeHost(hA);
    freeHost(hB);
}

TEST(GFOR, Inc_Scalar_Seq)
{
    const int num = 1000;
    const int st = 100;
    const int en = 500;
    const float val = 3;
    array A = randu(num);
    array B = A.copy();

    gfor(seq ii, st, en) {
        A(ii) += val;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();

    for (int i = 0; i < num; i++) {
        if (i >= st && i <= en) ASSERT_EQ(hA[i], hB[i] + val);
        else ASSERT_EQ(hA[i], hB[i]);
    }

    freeHost(hA);
    freeHost(hB);
}

TEST(GFOR, Assign_Array_Span)
{
    const int nx = 1000;
    array A = randu(nx);
    array B = randu(1, 1);

    gfor(seq ii, nx) {
        A(ii) = B;
    }

    float *hA = A.host<float>();
    float val = B.scalar<float>();

    for (int i = 0; i < nx; i++) {
        ASSERT_EQ(hA[i], val);
    }

    freeHost(hA);
}

TEST(GFOR, Assign_Array_Seq)
{
    const int nx = 1000;
    const int ny = 25;
    const int st = 100;
    const int en = 500;
    array A = randu(nx, ny);
    array B = A.copy();
    array C = randu(1, ny);

    gfor(seq ii, st, en) {
        A(ii, span) = C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        float val = hC[j];
        const int off = j * nx;
        for (int i = 0; i < nx; i++) {
            if (i >= st && i <= en) ASSERT_EQ(hA[i + off], val);
            else ASSERT_EQ(hA[i + off], hB[i + off]);
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hC);
}

TEST(GFOR, Inc_Array_Span)
{
    const int nx = 1000;
    array A = randu(nx);
    array B = A.copy();
    array C = randu(1, 1);

    gfor(seq ii, nx) {
        A(ii) += C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float val = C.scalar<float>();

    for (int i = 0; i < nx; i++) {
        ASSERT_EQ(hA[i], val + hB[i]);
    }

    freeHost(hA);
    freeHost(hB);
}

TEST(GFOR, Inc_Array_Seq)
{
    const int nx = 1000;
    const int ny = 25;
    const int st = 100;
    const int en = 500;
    array A = randu(nx, ny);
    array B = A.copy();
    array C = randu(1, ny);

    gfor(seq ii, st, en) {
        A(ii, span) += C;
    }

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        float val = hC[j];
        const int off = j * nx;
        for (int i = 0; i < nx; i++) {
            if (i >= st && i <= en) ASSERT_EQ(hA[i + off], val + hB[i + off]);
            else ASSERT_EQ(hA[i + off], hB[i + off]);
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hC);
}

TEST(BatchFunc, 2D0)
{
    const int nx = 1000;
    const int ny = 10;
    array A = randu(nx, ny);
    array B = randu( 1, ny);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(hA[j * nx + i] + hB[j], hC[j * nx + i]);
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 2D1)
{
    const int nx = 1000;
    const int ny = 10;
    array A = randu(nx, ny);
    array B = randu(nx, 1);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(hA[j * nx + i] + hB[i], hC[j * nx + i]);
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 3D0)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    array A = randu(nx, ny, nz);
    array B = randu( 1, ny, nz);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                ASSERT_EQ(hA[k * ny * nx + j * nx + i] + hB[k * ny + j], hC[k * ny * nx + j * nx + i]);
            }
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 3D1)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    array A = randu(nx, ny, nz);
    array B = randu(nx,  1, nz);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                ASSERT_EQ(hA[k * ny * nx + j * nx + i] + hB[k * nx + i], hC[k * ny * nx + j * nx + i]);
            }
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 3D2)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    array A = randu(nx, ny, nz);
    array B = randu(nx, ny,  1);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                ASSERT_EQ(hA[k * ny * nx + j * nx + i] + hB[j * nx + i], hC[k * ny * nx + j * nx + i]);
            }
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 3D01)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    array A = randu(nx, ny, nz);
    array B = randu( 1,  1, nz);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                ASSERT_EQ(hA[k * ny * nx + j * nx + i] + hB[k], hC[k * ny * nx + j * nx + i]);
            }
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 3D_1_2)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    array A = randu(nx, ny,  1);
    array B = randu(nx,  1, nz);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                ASSERT_EQ(hA[j * nx + i] + hB[k * nx + i], hC[k * ny * nx + j * nx + i]);
            }
        }
    }

    gforSet(false);
}

TEST(BatchFunc, 4D3)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    const int nw = 2;
    array A = randu(nx, ny, nz, nw);
    array B = randu(nx, ny, nz,  1);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int l = 0; l < nw; l++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ASSERT_EQ(hA[l * nz * ny * nx + k * ny * nx + j * nx + i] +
                              hB[k * ny * nx + j * nx + i],
                              hC[l * nz * ny * nx + k * ny * nx + j * nx + i]);
                }
            }
        }
    }

    gforSet(false);
}


TEST(BatchFunc, 4D_2_3)
{
    const int nx = 1000;
    const int ny = 10;
    const int nz = 3;
    const int nw = 2;
    array A = randu(nx,  1, nz, nw);
    array B = randu(nx, ny,  1,  1);

    gforSet(true);

    array C = A + B;

    float *hA = A.host<float>();
    float *hB = B.host<float>();
    float *hC = C.host<float>();

    for (int l = 0; l < nw; l++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ASSERT_EQ(hA[l * nz * nx + k * nx + i] +
                              hB[j * nx + i], hC[l * nz * ny * nx + k * ny * nx + j * nx + i]);
                }
            }
        }
    }

    gforSet(false);
}

TEST(ASSIGN, ISSUE_1127)
{
    using namespace af;
    array orig = randu(512, 768, 3);
    array vert = randu(512, 768, 3);
    array horiz = randu(512, 768, 3);
    array diag = randu(512, 768, 3);

    array out0 = constant(0, orig.dims(0) * 2, orig.dims(1) * 2, orig.dims(2));
    array out1 = constant(0, orig.dims(0) * 2, orig.dims(1) * 2, orig.dims(2));
    int rows = out0.dims(0), cols = out0.dims(1);

    gfor(seq chan, 3) {
        out0(seq(0,rows-1,2), seq(0,cols-1,2), chan) = orig(span,span,chan);
        out0(seq(1,rows-1,2), seq(0,cols-1,2), chan) = vert(span,span,chan);
        out0(seq(0,rows-1,2), seq(1,cols-1,2), chan) = horiz(span,span,chan);
        out0(seq(1,rows-1,2), seq(1,cols-1,2), chan) = diag(span,span,chan);
    }
    out1(seq(0,rows-1,2), seq(0,cols-1,2), span) = orig;
    out1(seq(1,rows-1,2), seq(0,cols-1,2), span) = vert;
    out1(seq(0,rows-1,2), seq(1,cols-1,2), span) = horiz;
    out1(seq(1,rows-1,2), seq(1,cols-1,2), span) = diag;

    std::vector<float> hout0(out0.elements());
    std::vector<float> hout1(out1.elements());

    out0.host(&hout0[0]);
    out1.host(&hout1[0]);

    for (int i = 0; i < out0.elements(); i++) {
        ASSERT_EQ(hout0[i], hout1[i]);
    }
}
