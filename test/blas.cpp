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
#include <af/blas.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/half.h>
#include <af/traits.hpp>
#include <algorithm>
#include <string>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dot;
using af::dtype_traits;
using af::getDevice;
using af::getDeviceCount;
using af::matmul;
using af::max;
using af::randu;
using af::setDevice;
using af::span;
using af::transpose;
using std::copy;
using std::cout;
using std::endl;
using std::ostream_iterator;
using std::string;
using std::stringstream;
using std::vector;

template<typename T>
class MatrixMultiply : public ::testing::Test {};

typedef ::testing::Types<float, double, cdouble, cfloat> TestTypes;
TYPED_TEST_SUITE(MatrixMultiply, TestTypes);

template<typename T, bool isBVector>
void MatMulCheck(string TestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T, T, int>(TestFile, numDims, hData, tests);

    af_array a, aT, b, bT;
    ASSERT_SUCCESS(af_create_array(&a, &hData[0].front(), numDims[0].ndims(),
                                   numDims[0].get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    dim4 atdims = numDims[0];
    {
        dim_t f   = atdims[0];
        atdims[0] = atdims[1];
        atdims[1] = f;
    }
    ASSERT_SUCCESS(af_moddims(&aT, a, atdims.ndims(), atdims.get()));
    ASSERT_SUCCESS(af_create_array(&b, &hData[1].front(), numDims[1].ndims(),
                                   numDims[1].get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    dim4 btdims = numDims[1];
    {
        dim_t f   = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }
    ASSERT_SUCCESS(af_moddims(&bT, b, btdims.ndims(), btdims.get()));

    vector<af_array> out(tests.size(), 0);
    if (isBVector) {
        ASSERT_SUCCESS(af_matmul(&out[0], aT, b, AF_MAT_NONE, AF_MAT_NONE));
        ASSERT_SUCCESS(af_matmul(&out[1], bT, a, AF_MAT_NONE, AF_MAT_NONE));
        ASSERT_SUCCESS(af_matmul(&out[2], b, a, AF_MAT_TRANS, AF_MAT_NONE));
        ASSERT_SUCCESS(af_matmul(&out[3], bT, aT, AF_MAT_NONE, AF_MAT_TRANS));
        ASSERT_SUCCESS(af_matmul(&out[4], b, aT, AF_MAT_TRANS, AF_MAT_TRANS));
    } else {
        ASSERT_SUCCESS(af_matmul(&out[0], a, b, AF_MAT_NONE, AF_MAT_NONE));
        ASSERT_SUCCESS(af_matmul(&out[1], a, bT, AF_MAT_NONE, AF_MAT_TRANS));
        ASSERT_SUCCESS(af_matmul(&out[2], a, bT, AF_MAT_TRANS, AF_MAT_NONE));
        ASSERT_SUCCESS(af_matmul(&out[3], aT, bT, AF_MAT_TRANS, AF_MAT_TRANS));
    }

    for (size_t i = 0; i < tests.size(); i++) {
        dim4 dd;
        dim_t *d = dd.get();
        af_get_dims(&d[0], &d[1], &d[2], &d[3], out[i]);
        ASSERT_VEC_ARRAY_NEAR(tests[i], dd, out[i], 1e-3);
    }

    ASSERT_SUCCESS(af_release_array(a));
    ASSERT_SUCCESS(af_release_array(aT));
    ASSERT_SUCCESS(af_release_array(b));
    ASSERT_SUCCESS(af_release_array(bT));

    for (size_t i = 0; i < out.size(); i++) {
        ASSERT_SUCCESS(af_release_array(out[i]));
    }
}

TYPED_TEST(MatrixMultiply, Square) {
    MatMulCheck<TypeParam, false>(TEST_DIR "/blas/Basic.test");
}

TYPED_TEST(MatrixMultiply, NonSquare) {
    MatMulCheck<TypeParam, false>(TEST_DIR "/blas/NonSquare.test");
}

TYPED_TEST(MatrixMultiply, SquareVector) {
    MatMulCheck<TypeParam, true>(TEST_DIR "/blas/SquareVector.test");
}

TYPED_TEST(MatrixMultiply, RectangleVector) {
    MatMulCheck<TypeParam, true>(TEST_DIR "/blas/RectangleVector.test");
}

template<typename T, bool isBVector>
void cppMatMulCheck(string TestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T, T, int>(TestFile, numDims, hData, tests);

    array a(numDims[0], &hData[0].front());
    array b(numDims[1], &hData[1].front());

    dim4 atdims = numDims[0];
    {
        dim_t f   = atdims[0];
        atdims[0] = atdims[1];
        atdims[1] = f;
    }
    dim4 btdims = numDims[1];
    {
        dim_t f   = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }

    array aT = moddims(a, atdims.ndims(), atdims.get());
    array bT = moddims(b, btdims.ndims(), btdims.get());

    vector<array> out(tests.size());
    if (isBVector) {
        out[0] = matmul(aT, b, AF_MAT_NONE, AF_MAT_NONE);
        out[1] = matmul(bT, a, AF_MAT_NONE, AF_MAT_NONE);
        out[2] = matmul(b, a, AF_MAT_TRANS, AF_MAT_NONE);
        out[3] = matmul(bT, aT, AF_MAT_NONE, AF_MAT_TRANS);
        out[4] = matmul(b, aT, AF_MAT_TRANS, AF_MAT_TRANS);
    } else {
        out[0] = matmul(a, b, AF_MAT_NONE, AF_MAT_NONE);
        out[1] = matmul(a, bT, AF_MAT_NONE, AF_MAT_TRANS);
        out[2] = matmul(a, bT, AF_MAT_TRANS, AF_MAT_NONE);
        out[3] = matmul(aT, bT, AF_MAT_TRANS, AF_MAT_TRANS);
    }

    for (size_t i = 0; i < tests.size(); i++) {
        dim_t elems = out[i].elements();
        vector<T> h_out(elems);
        out[i].host((void *)&h_out.front());

        if (false == equal(h_out.begin(), h_out.end(), tests[i].begin())) {
            cout << "Failed test " << i << "\nCalculated: " << endl;
            copy(h_out.begin(), h_out.end(), ostream_iterator<T>(cout, ", "));
            cout << "Expected: " << endl;
            copy(tests[i].begin(), tests[i].end(),
                 ostream_iterator<T>(cout, ", "));
            FAIL();
        }
    }
}

TYPED_TEST(MatrixMultiply, Square_CPP) {
    cppMatMulCheck<TypeParam, false>(TEST_DIR "/blas/Basic.test");
}

TYPED_TEST(MatrixMultiply, NonSquare_CPP) {
    cppMatMulCheck<TypeParam, false>(TEST_DIR "/blas/NonSquare.test");
}

TYPED_TEST(MatrixMultiply, SquareVector_CPP) {
    cppMatMulCheck<TypeParam, true>(TEST_DIR "/blas/SquareVector.test");
}

TYPED_TEST(MatrixMultiply, RectangleVector_CPP) {
    cppMatMulCheck<TypeParam, true>(TEST_DIR "/blas/RectangleVector.test");
}

#define DEVICE_ITERATE(func)                             \
    do {                                                 \
        const char *ENV = getenv("AF_MULTI_GPU_TESTS");  \
        if (ENV && ENV[0] == '0') {                      \
            func;                                        \
        } else {                                         \
            int oldDevice = getDevice();                 \
            for (int i = 0; i < getDeviceCount(); i++) { \
                setDevice(i);                            \
                func;                                    \
            }                                            \
            setDevice(oldDevice);                        \
        }                                                \
    } while (0);

TYPED_TEST(MatrixMultiply, MultiGPUSquare_CPP) {
    DEVICE_ITERATE(
        (cppMatMulCheck<TypeParam, false>(TEST_DIR "/blas/Basic.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPUNonSquare_CPP) {
    DEVICE_ITERATE(
        (cppMatMulCheck<TypeParam, false>(TEST_DIR "/blas/NonSquare.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPUSquareVector_CPP) {
    DEVICE_ITERATE(
        (cppMatMulCheck<TypeParam, true>(TEST_DIR "/blas/SquareVector.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPURectangleVector_CPP) {
    DEVICE_ITERATE((cppMatMulCheck<TypeParam, true>(
        TEST_DIR "/blas/RectangleVector.test")));
}

float batch_tol = 1E-2;
TEST(MatrixMultiply, Batched) {
    const int M  = 512;
    const int K  = 512;
    const int N  = 10;
    const int D2 = 2;
    const int D3 = 3;
    for (int d3 = 1; d3 <= D3; d3 *= D3) {
        for (int d2 = 1; d2 <= D2; d2 *= D2) {
            array a = randu(M, K, d2, d3);
            array b = randu(K, N, d2, d3);
            array c = matmul(a, b);

            for (int j = 0; j < d3; j++) {
                for (int i = 0; i < d2; i++) {
                    array a_ij = a(span, span, i, j);
                    array b_ij = b(span, span, i, j);
                    array c_ij = c(span, span, i, j);
                    array res  = matmul(a_ij, b_ij);
                    ASSERT_ARRAYS_NEAR(c_ij, res, batch_tol);
                }
            }
        }
    }
}

#undef DEVICE_ITERATE

TEST(MatrixMultiply, ISSUE_1882) {
    const int m = 2;
    const int n = 3;
    array A     = randu(m, n);
    array BB    = randu(n, m);
    array B     = BB(0, span);

    array res1 = matmul(A.T(), B.T());
    array res2 = matmulTT(A, B);

    ASSERT_ARRAYS_NEAR(res1, res2, 1E-5);
}

struct blas_params {
    int m, n, k, ld2, ld3, rd2, rd3;
    af_dtype type;
    blas_params(int m_, int n_, int k_, int ld2_, int ld3_, int rd2_, int rd3_,
                af_dtype type_)
        : m(m_)
        , n(n_)
        , k(k_)
        , ld2(ld2_)
        , ld3(ld3_)
        , rd2(rd2_)
        , rd3(rd3_)
        , type(type_) {}
};

class MatrixMultiplyBatch : public ::testing::TestWithParam<blas_params> {
   public:
    array lhs, rhs, out, gold;
    void SetUp() {
        blas_params params = GetParam();
        lhs = randu(params.m, params.k, params.ld2, params.ld3, params.type);
        rhs = randu(params.k, params.n, params.rd2, params.rd3, params.type);

        gold = array(params.m, params.n, std::max(params.ld2, params.rd2),
                     std::max(params.ld3, params.rd3));

        if (params.ld2 == params.rd2 && params.ld3 == params.rd3) {
            for (int i = 0; i < params.ld2; i++) {
                for (int j = 0; j < params.ld3; j++) {
                    array lhs_sub          = lhs(span, span, i, j);
                    array rhs_sub          = rhs(span, span, i, j);
                    gold(span, span, i, j) = matmul(lhs_sub, rhs_sub);
                }
            }
        } else {
            for (int i = 0; i < params.ld2; i++) {
                for (int j = 0; j < params.ld3; j++) {
                    for (int k = 0; k < params.rd2; k++) {
                        for (int l = 0; l < params.rd3; l++) {
                            array lhs_sub = lhs(span, span, i, j);
                            array rhs_sub = rhs(span, span, k, l);
                            gold(span, span, std::max(i, k), std::max(j, l)) =
                                matmul(lhs_sub, rhs_sub);
                        }
                    }
                }
            }
        }
    }
};

std::string print_blas_params(
    const ::testing::TestParamInfo<MatrixMultiplyBatch::ParamType> info) {
    std::stringstream ss;

    ss << "LHS_" << info.param.m << "x" << info.param.k << "x" << info.param.ld2
       << "x" << info.param.ld3 << "__RHS" << info.param.k << "x"
       << info.param.n << "x" << info.param.rd2 << "x" << info.param.rd3;

    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    LHSBroadcast, MatrixMultiplyBatch,
    ::testing::Values(

        // clang-format off
            //             M      N     K   ld2  ld3   rd2   rd3  type
            blas_params( 32,     32,   10,    2,   1,    1,    1,  f32),
            blas_params( 32,     32,   10,    1,   2,    1,    1,  f32),
            blas_params( 32,     32,   10,    2,   2,    1,    1,  f32),
            blas_params( 32,     32,   10,    3,   2,    1,    1,  f32),
            blas_params( 32,     32,   10,    3,   3,    1,    1,  f32),
            blas_params( 32,     32,   10,    4,   4,    1,    1,  f32),

            blas_params(512,     32,  512,    4,   4,    1,    1,  f32),
            blas_params(512,     32,  513,    4,   4,    1,    1,  f32),
            blas_params(513,     32,  513,    4,   4,    1,    1,  f32),
            blas_params(513,     33,  513,    4,   4,    1,    1,  f32),
            blas_params(513,    511,   32,    4,   4,    1,    1,  f32),
            blas_params(513,    511,   31,    4,   4,    1,    1,  f32),
            blas_params(513,    511,   33,    4,   4,    1,    1,  f32),
            blas_params(511,    511,   33,    4,   4,    1,    1,  f32)
        // clang-format on

        ),
    print_blas_params);

INSTANTIATE_TEST_SUITE_P(
    RHSBroadcast, MatrixMultiplyBatch,
    ::testing::Values(
        // clang-format off
            //            M      N     K   ld2  ld3   rd2  rd3  type
            blas_params( 32 ,    32,  10,    1,   1,    2,   1,  f32),
            blas_params( 32 ,    32,  10,    1,   1,    1,   2,  f32),
            blas_params( 32 ,    32,  10,    1,   1,    2,   2,  f32),
            blas_params( 32 ,    32,  10,    1,   1,    3,   2,  f32),
            blas_params( 32 ,    32,  10,    1,   1,    3,   3,  f32),
            blas_params( 32 ,    32,  10,    1,   1,    4,   4,  f32),

            blas_params(512 ,    32,  512,   1,   1,    4,   4,  f32),
            blas_params(512 ,    32,  513,   1,   1,    4,   4,  f32),
            blas_params(513 ,    32,  513,   1,   1,    4,   4,  f32),
            blas_params(513 ,    33,  513,   1,   1,    4,   4,  f32),
            blas_params(513 ,   511,   32,   1,   1,    4,   4,  f32),
            blas_params(513 ,   511,   31,   1,   1,    4,   4,  f32),
            blas_params(513 ,   511,   33,   1,   1,    4,   4,  f32),
            blas_params(511 ,   511,   33,   1,   1,    4,   4,  f32)
        // clang-format on
        ),
    print_blas_params);

INSTANTIATE_TEST_SUITE_P(
    SameBatch, MatrixMultiplyBatch,
    ::testing::Values(
        // clang-format off
            //          M      N     K   ld2  ld3   rd2  rd3  type
            blas_params(32,   32,  10,     2,   1,    2,   1,  f32),
            blas_params(32,   32,  10,     1,   2,    1,   2,  f32),
            blas_params(32,   32,  10,     2,   2,    2,   2,  f32),
            blas_params(32,   32,  10,     3,   2,    3,   2,  f32),
            blas_params(32,   32,  10,     3,   3,    3,   3,  f32),
            blas_params(32,   32,  10,     4,   4,    4,   4,  f32),

            blas_params(512,  32, 512,     4,   4,    4,   4,  f32),
            blas_params(512,  32, 513,     4,   4,    4,   4,  f32),
            blas_params(513,  32, 513,     4,   4,    4,   4,  f32),
            blas_params(513,  33, 513,     4,   4,    4,   4,  f32),
            blas_params(513, 511,  32,     4,   4,    4,   4,  f32),
            blas_params(513, 511,  31,     4,   4,    4,   4,  f32),
            blas_params(513, 511,  33,     4,   4,    4,   4,  f32),
            blas_params(511, 511,  33,     4,   4,    4,   4,  f32),

            blas_params( 32,  32,  10,     1,   1,    1,   1, f32)
        // clang-format on
        ),
    print_blas_params);

TEST_P(MatrixMultiplyBatch, Batched) {
    array out = matmul(lhs, rhs);
    ASSERT_ARRAYS_NEAR(gold, out, 1e-3);
}

float alpha = 1.f;
float beta  = 0.f;

float h_gold_gemv[4]  = {5, 5, 5, 5};
float h_half_ones[20] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                         1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

float h_lhs[9] = {1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f};

float h_lhs_tall[6] = {1.f, 3.f, 5.f, 2.f, 4.f, 6.f};

float h_lhs_wide[6] = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};

float h_lhs_batch[18] = {1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f,

                         8.f, 2.f, 5.f, 3.f, 4.f, 7.f, 1.f, 0.f, 6.f};

float h_rhs[9] = {9.f, 6.f, 3.f, 8.f, 5.f, 2.f, 7.f, 4.f, 1.f};

float h_rhs_tall[6] = {9.f, 7.f, 5.f, 8.f, 6.f, 4.f};

float h_rhs_wide[6] = {9.f, 6.f, 8.f, 5.f, 7.f, 4.f};

float h_gold[9] = {30.f, 84.f, 138.f, 24.f, 69.f, 114.f, 18.f, 54.f, 90.f};

float h_gold_NN[9] = {21.f, 51.f, 81.f, 18.f, 44.f, 70.f, 15.f, 37.f, 59.f};

float h_gold_NT[9] = {25.f, 59.f, 93.f, 19.f, 45.f, 71.f, 13.f, 31.f, 49.f};

float h_gold_TN[4] = {55.f, 76.f, 46.f, 64.f};

float h_gold_TT[4] = {68.f, 92.f, 41.f, 56.f};

float h_gold_batch[18] = {
    30.f, 84.f, 138.f, 24.f, 69.f, 114.f, 18.f, 54.f, 90.f,

    93.f, 42.f, 105.f, 81.f, 36.f, 87.f,  69.f, 30.f, 69.f};

TEST(MatrixMultiply, float) {
    array A32           = array(3, 3, h_lhs);
    array B32           = array(3, 3, h_rhs);
    af_array C32        = 0;
    const float alpha32 = 1.0f;
    const float beta32  = 0.0f;
    af_gemm(&C32, AF_MAT_NONE, AF_MAT_NONE, &alpha32, A32.get(), B32.get(),
            &beta32);
    array expected32 = array(3, 3, h_gold);
    ASSERT_ARRAYS_NEAR(expected32, af::array(C32), 0.0001);
}

TEST(MatrixMultiply, half) {
    SUPPORTED_TYPE_CHECK(af_half);

    array A16        = array(3, 3, h_lhs).as(f16);
    array B16        = array(3, 3, h_rhs).as(f16);
    array expected16 = array(3, 3, h_gold).as(f16);

    {
        af_array C16 = 0;
        const half_float::half alpha16(1.0f);
        const half_float::half beta16(0.0f);
        ASSERT_SUCCESS(af_gemm(&C16, AF_MAT_NONE, AF_MAT_NONE, &alpha16,
                               A16.get(), B16.get(), &beta16));
        af::array C(C16);
        ASSERT_ARRAYS_NEAR(expected16, C, 0.00001);
    }
    {
        array C16 = matmul(A16, B16);
        ASSERT_ARRAYS_NEAR(expected16, C16, 0.000001);
    }
}

struct test_params {
    af_mat_prop opt_lhs;
    af_mat_prop opt_rhs;
    float *alpha;
    float *h_lhs;
    float *h_rhs;
    float *h_gold;
    dim4 lhs_dims;
    dim4 rhs_dims;
    dim4 out_dims;
    float *beta;
    TestOutputArrayType out_array_type;

    test_params(af_mat_prop optl, af_mat_prop optr, float *a, float *l,
                float *r, float *g, dim4 ldims, dim4 rdims, dim4 odims,
                float *b, TestOutputArrayType t)
        : opt_lhs(optl)
        , opt_rhs(optr)
        , alpha(a)
        , h_lhs(l)
        , h_rhs(r)
        , h_gold(g)
        , lhs_dims(ldims)
        , rhs_dims(rdims)
        , out_dims(odims)
        , beta(b)
        , out_array_type(t) {}
};

class Gemm : public ::testing::TestWithParam<test_params> {
   protected:
    af_array lhs;
    af_array rhs;
    af_array gold;
    af_array out;
    TestOutputArrayInfo metadata;

    void SetUp() {
        test_params params = GetParam();

        lhs  = 0;
        rhs  = 0;
        out  = 0;
        gold = 0;

        ASSERT_SUCCESS(af_create_array(&lhs, params.h_lhs,
                                       params.lhs_dims.ndims(),
                                       params.lhs_dims.get(), f32));
        ASSERT_SUCCESS(af_create_array(&rhs, params.h_rhs,
                                       params.rhs_dims.ndims(),
                                       params.rhs_dims.get(), f32));

        dim_t gold_dim0 = params.opt_lhs == AF_MAT_TRANS ? params.lhs_dims[1]
                                                         : params.lhs_dims[0];
        dim_t gold_dim1 = params.opt_rhs == AF_MAT_TRANS ? params.rhs_dims[0]
                                                         : params.rhs_dims[1];
        dim_t gold_dim2 = std::max(params.lhs_dims[2], params.rhs_dims[2]);
        dim_t gold_dim3 = std::max(params.lhs_dims[3], params.rhs_dims[3]);
        dim4 gold_dims(gold_dim0, gold_dim1, gold_dim2, gold_dim3);

        metadata = TestOutputArrayInfo(params.out_array_type);
        genTestOutputArray(&out, params.out_dims.ndims(), params.out_dims.get(),
                           f32, &metadata);

        ASSERT_SUCCESS(af_create_array(&gold, params.h_gold, gold_dims.ndims(),
                                       gold_dims.get(), f32));
    }

    void TearDown() {
        if (gold != 0) { ASSERT_SUCCESS(af_release_array(gold)); }
        if (rhs != 0) { ASSERT_SUCCESS(af_release_array(rhs)); }
        if (lhs != 0) { ASSERT_SUCCESS(af_release_array(lhs)); }
    }
};

void replace_all(std::string &str, const std::string &oldStr,
                 const std::string &newStr) {
    std::string::size_type pos = 0u;
    while ((pos = str.find(oldStr, pos)) != std::string::npos) {
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
}

std::string concat_dim4(dim4 d) {
    std::stringstream ss;
    ss << d;
    std::string s = ss.str();
    replace_all(s, " ", "x");
    return s;
}

string out_info(const ::testing::TestParamInfo<Gemm::ParamType> info) {
    test_params params = info.param;

    stringstream ss;
    switch (params.out_array_type) {
        case NULL_ARRAY: ss << "NullOut"; break;
        case FULL_ARRAY: ss << "FullOut"; break;
        case SUB_ARRAY: ss << "SubarrayOut"; break;
        case REORDERED_ARRAY: ss << "ReorderedOut"; break;
        default: ss << "UnknownOutArrayType"; break;
    }

    ss << "_" << concat_dim4(params.lhs_dims) << "_"
       << concat_dim4(params.rhs_dims);

    ss << "_";
    ss << (params.opt_lhs == AF_MAT_TRANS ? "T" : "N");
    ss << (params.opt_rhs == AF_MAT_TRANS ? "T" : "N");

    if (params.lhs_dims[2] > 1 || params.rhs_dims[2] > 1) { ss << "_Batched"; }

    return ss.str();
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Square, Gemm,
    ::testing::Values(
        //          lhs_opts     rhs_opts     alpha  lhs    rhs    gold    lhs_dims    rhs_dims    out_dims    beta  out_array_type
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs, h_rhs, h_gold, dim4(3, 3), dim4(3, 3), dim4(3, 3), &beta, NULL_ARRAY     ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs, h_rhs, h_gold, dim4(3, 3), dim4(3, 3), dim4(3, 3), &beta, FULL_ARRAY     ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs, h_rhs, h_gold, dim4(3, 3), dim4(3, 3), dim4(3, 3), &beta, SUB_ARRAY      ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs, h_rhs, h_gold, dim4(3, 3), dim4(3, 3), dim4(3, 3), &beta, REORDERED_ARRAY)
        ),
    out_info
    );
// clang-format on

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Batched, Gemm,
    ::testing::Values(
        //          lhs_opts     rhs_opts     alpha  lhs          rhs    gold          lhs_dims       rhs_dims    out_dims       beta  out_array_type
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs_batch, h_rhs, h_gold_batch, dim4(3, 3, 2), dim4(3, 3), dim4(3, 3, 2), &beta, NULL_ARRAY     ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs_batch, h_rhs, h_gold_batch, dim4(3, 3, 2), dim4(3, 3), dim4(3, 3, 2), &beta, FULL_ARRAY     ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs_batch, h_rhs, h_gold_batch, dim4(3, 3, 2), dim4(3, 3), dim4(3, 3, 2), &beta, SUB_ARRAY      ),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_lhs_batch, h_rhs, h_gold_batch, dim4(3, 3, 2), dim4(3, 3), dim4(3, 3, 2), &beta, REORDERED_ARRAY)
        ),
    out_info
    );
// clang-format on

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    NonSquare, Gemm,
    ::testing::Values(
        //          lhs_opts      rhs_opts      alpha  lhs         rhs         gold       lhs_dims    rhs_dims    out_dims    beta  out_array_type
        test_params(AF_MAT_NONE,  AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_wide, h_gold_NN, dim4(3, 2), dim4(2, 3), dim4(3, 3), &beta, NULL_ARRAY),
        test_params(AF_MAT_NONE,  AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_tall, h_gold_NT, dim4(3, 2), dim4(3, 2), dim4(3, 3), &beta, NULL_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_tall, h_gold_TN, dim4(3, 2), dim4(3, 2), dim4(2, 2), &beta, NULL_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_wide, h_gold_TT, dim4(3, 2), dim4(2, 3), dim4(2, 2), &beta, NULL_ARRAY),

        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_half_ones, h_half_ones, h_gold_gemv, dim4(4, 5), dim4(5, 1), dim4(4, 1), &beta, NULL_ARRAY),
        test_params(AF_MAT_NONE, AF_MAT_NONE, &alpha, h_half_ones, h_half_ones, h_gold_gemv, dim4(1, 5), dim4(5, 1), dim4(1, 1), &beta, NULL_ARRAY),
        test_params(AF_MAT_NONE, AF_MAT_TRANS, &alpha, h_half_ones, h_half_ones, h_gold_gemv, dim4(4, 5), dim4(1, 5), dim4(4, 1), &beta, NULL_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_NONE, &alpha, h_half_ones, h_half_ones, h_gold_gemv, dim4(5, 4), dim4(5, 1), dim4(4, 1), &beta, NULL_ARRAY),

        test_params(AF_MAT_NONE,  AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_wide, h_gold_NN, dim4(3, 2), dim4(2, 3), dim4(3, 3), &beta, FULL_ARRAY),
        test_params(AF_MAT_NONE,  AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_tall, h_gold_NT, dim4(3, 2), dim4(3, 2), dim4(3, 3), &beta, FULL_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_tall, h_gold_TN, dim4(3, 2), dim4(3, 2), dim4(2, 2), &beta, FULL_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_wide, h_gold_TT, dim4(3, 2), dim4(2, 3), dim4(2, 2), &beta, FULL_ARRAY),

        test_params(AF_MAT_NONE,  AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_wide, h_gold_NN, dim4(3, 2), dim4(2, 3), dim4(3, 3), &beta, SUB_ARRAY),
        test_params(AF_MAT_NONE,  AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_tall, h_gold_NT, dim4(3, 2), dim4(3, 2), dim4(3, 3), &beta, SUB_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_tall, h_gold_TN, dim4(3, 2), dim4(3, 2), dim4(2, 2), &beta, SUB_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_wide, h_gold_TT, dim4(3, 2), dim4(2, 3), dim4(2, 2), &beta, SUB_ARRAY),

        test_params(AF_MAT_NONE,  AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_wide, h_gold_NN, dim4(3, 2), dim4(2, 3), dim4(3, 3), &beta, REORDERED_ARRAY),
        test_params(AF_MAT_NONE,  AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_tall, h_gold_NT, dim4(3, 2), dim4(3, 2), dim4(3, 3), &beta, REORDERED_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_NONE,  &alpha, h_lhs_tall, h_rhs_tall, h_gold_TN, dim4(3, 2), dim4(3, 2), dim4(2, 2), &beta, REORDERED_ARRAY),
        test_params(AF_MAT_TRANS, AF_MAT_TRANS, &alpha, h_lhs_tall, h_rhs_wide, h_gold_TT, dim4(3, 2), dim4(2, 3), dim4(2, 2), &beta, REORDERED_ARRAY)
        ),
    out_info
    );
// clang-format on

TEST_P(Gemm, UsePreallocatedOutArray) {
    test_params params = GetParam();
    ASSERT_SUCCESS(af_gemm(&out, params.opt_lhs, params.opt_rhs, params.alpha,
                           lhs, rhs, params.beta));

    ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
}

TEST(Gemm, DocSnippet) {
    //! [ex_af_gemm_alloc]
    af_array A, B;

    dim_t adims[] = {5, 3, 2};
    dim_t bdims[] = {3, 5, 2};
    af_constant(&A, 1, 3, adims, f32);
    af_constant(&B, 1, 3, bdims, f32);

    float alpha = 1.f;
    float beta  = 0.f;

    // Undefined behavior!
    // af_array undef;
    // af_gemm(&undef, AF_MAT_NONE, AF_MAT_NONE, &alpha, a.get(), b.get(),
    // &beta);

    af_array C = 0;
    af_gemm(&C, AF_MAT_NONE, AF_MAT_NONE, &alpha, A, B, &beta);
    // C =
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.

    //! [ex_af_gemm_alloc]

    af_array c1_copy = 0;
    ASSERT_SUCCESS(af_retain_array(&c1_copy, C));
    af::array c1(c1_copy);
    af::array gold1 = af::constant(3, 5, 5, 2, f32);
    ASSERT_ARRAYS_EQ(gold1, c1);

    //! [ex_af_gemm_overwrite]
    alpha                = 1.f;
    beta                 = 1.f;
    af_seq first_slice[] = {af_span, af_span, {0., 0., 1.}};
    af_array Asub, Bsub, Csub;
    af_index(&Asub, A, 3, first_slice);
    af_index(&Bsub, B, 3, first_slice);
    af_index(&Csub, C, 3, first_slice);
    af_gemm(&Csub, AF_MAT_NONE, AF_MAT_NONE, &alpha, Asub, Bsub, &beta);
    // C =
    //  6.   6.   6.   6.   6.
    //  6.   6.   6.   6.   6.
    //  6.   6.   6.   6.   6.
    //  6.   6.   6.   6.   6.
    //  6.   6.   6.   6.   6.
    //
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //  3.   3.   3.   3.   3.
    //! [ex_af_gemm_overwrite]

    af_array c2_copy = 0;
    ASSERT_SUCCESS(af_retain_array(&c2_copy, C));
    af::array c2(c2_copy);
    vector<float> gold2(5 * 5 * 2, 3);
    fill(gold2.begin(), gold2.begin() + (5 * 5), 6);

    af_release_array(A);
    af_release_array(B);
    af_release_array(C);
    af_release_array(Asub);
    af_release_array(Bsub);
    af_release_array(Csub);

    ASSERT_VEC_ARRAY_EQ(gold2, dim4(5, 5, 2), c2);
}

TEST(Gemv, HalfScalarProduct) {
    SUPPORTED_TYPE_CHECK(half_float::half);

    const unsigned int sizeValue = 5;
    array gold                   = constant(sizeValue, 4, 1, f16);
    {
        array a     = constant(1, 4, sizeValue, f16);
        array b     = constant(1, sizeValue, 1, f16);
        array mmRes = matmul(a, b);
        ASSERT_ARRAYS_EQ(mmRes, gold);
    }
    {
        array a      = constant(1, 1, sizeValue, f16);
        array b      = constant(1, sizeValue, 1, f16);
        array mmRes  = matmul(a, b);
        array dotRes = dot(transpose(a), b);
        ASSERT_ARRAYS_EQ(mmRes, dotRes);
    }
}

TEST(MatrixMultiply, SameInput) {
    // Tests for an error that occured in the Intel OpenCL GPU implementation
    // that caused an error when you passed the same array as the lhs and the
    // rhs. see #1711 and PR #2774. Caused by mapping the same buffer with
    // CL_MEM_WRITE access
    int dim = 10;
    array a = randu(dim, dim);
    vector<float> ha(dim * dim);
    a.host(&ha.front());

    vector<float> hgold(dim * dim, 0);

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                hgold[i * dim + j] += ha[k * dim + j] * ha[i * dim + k];
            }
        }
    }
    array out = matmul(a, a);
    ASSERT_VEC_ARRAY_NEAR(hgold, dim4(dim, dim), out, 1e-4);
}
