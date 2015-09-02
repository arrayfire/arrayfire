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
#include <algorithm>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::array;
using af::cfloat;
using af::cdouble;


template<typename T>
class Reduce : public ::testing::Test
{
};

typedef ::testing::Types<float, double, af::cfloat, af::cdouble, uint, int, intl, uintl, uchar> TestTypes;
TYPED_TEST_CASE(Reduce, TestTypes);

typedef af_err (*reduceFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, reduceFunc af_reduce>
void reduceTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_EQ(AF_SUCCESS, af_release_array(tempArray));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_reduce(&outArray, inArray, d + off));

        af_dtype t;
        af_get_type(&t, outArray);

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        if(std::equal(currGoldBar.begin(), currGoldBar.end(), outData) == false)
        {
            for (size_t elIter = 0; elIter < nElems; ++elIter) {

                EXPECT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                                << " for dim " << d + off << std::endl;
            }
            af_print_array(outArray);
            for(int i = 0; i < (int)nElems; i++) {
                cout << currGoldBar[i] << ", ";
            }

            cout << endl;
            for(int i = 0; i < (int)nElems; i++) {
                cout << outData[i] << ", ";
            }
            FAIL();
        }


        // Delete
        delete[] outData;
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

vector<af_seq> init_subs()
{
    vector<af_seq> subs;
    subs.push_back(af_make_seq(2, 6, 1));
    subs.push_back(af_make_seq(1, 5, 1));
    subs.push_back(af_make_seq(1, 3, 1));
    subs.push_back(af_make_seq(1, 2, 1));
    return subs;
}

template<typename T,reduceFunc OP>
struct promote_type {
    typedef T type;
};

// char and uchar are promoted to int for sum and product
template<> struct promote_type<uchar, af_sum>       { typedef uint type; };
template<> struct promote_type<char , af_sum>       { typedef uint type; };
template<> struct promote_type<uchar, af_product>   { typedef uint type; };
template<> struct promote_type<char , af_product>   { typedef uint type; };

#define REDUCE_TESTS(FN)                                                                    \
    TYPED_TEST(Reduce,Test_##FN)                                                    \
    {                                                                                       \
        reduceTest<TypeParam, typename promote_type<TypeParam, af_##FN>::type, af_##FN>(    \
            string(TEST_DIR"/reduce/"#FN".test")                                            \
            );                                                                              \
    }                                                                                       \

REDUCE_TESTS(sum);
REDUCE_TESTS(min);
REDUCE_TESTS(max);

#undef REDUCE_TESTS
#define REDUCE_TESTS(FN, OT)                        \
    TYPED_TEST(Reduce,Test_##FN)            \
    {                                               \
        reduceTest<TypeParam, OT, af_##FN>(         \
            string(TEST_DIR"/reduce/"#FN".test")    \
            );                                      \
    }                                               \

REDUCE_TESTS(any_true, unsigned char);
REDUCE_TESTS(all_true, unsigned char);
REDUCE_TESTS(count, unsigned);

#undef REDUCE_TESTS

TEST(Reduce,Test_Reduce_Big0)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big0.test"),
        0
        );
}

TEST(Reduce,Test_Reduce_Big1)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big1.test"),
        1
        );
}

/////////////////////////////////// CPP //////////////////////////////////
//
typedef af::array (*ReductionOp)(const af::array&, const int);

using af::sum;
using af::min;
using af::max;
using af::allTrue;
using af::anyTrue;
using af::count;

template<typename Ti, typename To, ReductionOp reduce>
void cppReduceTest(string pTestFile)
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af::array input(dims, &in.front());

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        af::array output = reduce(input, d);

        // Get result
        To *outData = new To[dims.elements()];
        output.host((void*)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for dim " << d << std::endl;
        }

        // Delete
        delete[] outData;
    }
}

#define CPP_REDUCE_TESTS(FN, FNAME, Ti, To)        \
    TEST(Reduce, Test_##FN##_CPP)                  \
    {                                              \
        cppReduceTest<Ti, To, FN>(                 \
            string(TEST_DIR"/reduce/"#FNAME".test")\
            );                                     \
    }

CPP_REDUCE_TESTS(sum, sum, float, float);
CPP_REDUCE_TESTS(min, min, float, float);
CPP_REDUCE_TESTS(max, max, float, float);
CPP_REDUCE_TESTS(anyTrue, any_true, float, unsigned char);
CPP_REDUCE_TESTS(allTrue, all_true, float, unsigned char);
CPP_REDUCE_TESTS(count, count, float, unsigned);

TEST(Reduce, Test_Product_Global)
{
    int num = 100;
    af::array a = 1 + af::round(5 * af::randu(num, 1)) / 100;

    float res = af::product<float>(a);
    float *h_a = a.host<float>();
    float gold = 1;

    for (int i = 0; i < num; i++) {
        gold *= h_a[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_Sum_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));

    float res = af::sum<float>(a);
    float *h_a = a.host<float>();
    float gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_a[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_Count_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));
    af::array b = a.as(b8);

    int res = af::count<int>(b);
    char *h_b = b.host<char>();
    int gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_b[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_b;
}

TEST(Reduce, Test_min_Global)
{
    if (noDoubleTests<double>()) return;

    int num = 10000;
    af::array a = af::randu(num, 1, f64);
    double res = af::min<double>(a);
    double *h_a = a.host<double>();
    double gold = std::numeric_limits<double>::max();

    if (noDoubleTests<double>()) return;

    for (int i = 0; i < num; i++) {
        gold = std::min(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_max_Global)
{
    int num = 10000;
    af::array a = af::randu(num, 1);
    float res = af::max<float>(a);
    float *h_a = a.host<float>();
    float gold = -std::numeric_limits<float>::max();

    for (int i = 0; i < num; i++) {
        gold = std::max(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}


template<typename T>
void typed_assert_eq(T lhs, T rhs, bool both = true)
{
    ASSERT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<float>(float lhs, float rhs, bool both)
{
    ASSERT_FLOAT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<double>(double lhs, double rhs, bool both)
{
    ASSERT_DOUBLE_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<af::cfloat>(af::cfloat lhs, af::cfloat rhs, bool both)
{
    ASSERT_FLOAT_EQ(real(lhs), real(rhs));
    if(both)
        ASSERT_FLOAT_EQ(imag(lhs), imag(rhs));

}

template<>
void typed_assert_eq<af::cdouble>(af::cdouble lhs, af::cdouble rhs, bool both)
{
    ASSERT_DOUBLE_EQ(real(lhs), real(rhs));
    if(both)
        ASSERT_DOUBLE_EQ(imag(lhs), imag(rhs));
}

TYPED_TEST(Reduce, Test_All_Global)
{
    if (noDoubleTests<TypeParam>()) return;

    // Input size test
    for(int i = 1; i < 1000; i+=100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam)true);
        array a(2, num/2, &h_vals.front());

        TypeParam res = af::allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);

        h_vals[3] = false;
        a = array(2, num/2, &h_vals.front());

        res = af::allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);
    }

    // false value location test
    int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam)true);
    for(int i = 1; i < 10000; i+=100) {
        h_vals[i] = false;
        array a(2, num/2, &h_vals.front());

        TypeParam res = af::allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);

        h_vals[i] = true;
    }
}

TYPED_TEST(Reduce, Test_Any_Global)
{
    if (noDoubleTests<TypeParam>()) return;

    // Input size test
    for(int i = 1; i < 1000; i+=100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam)false);
        array a(2, num/2, &h_vals.front());

        TypeParam res = af::anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)false, res, false);

        h_vals[3] = true;
        a = array(2, num/2, &h_vals.front());

        res = af::anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);
    }

    // true value location test
    int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam)false);
    for(int i = 1; i < 10000; i+=100) {
        h_vals[i] = true;
        array a(2, num/2, &h_vals.front());

        TypeParam res = af::anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam)true, res, false);

        h_vals[i] = false;
    }
}

TEST(MinMax, NaN)
{
    const int num = 10000;
    af::array A = af::randu(num);
    A(where(A < 0.25)) = af::NaN;

    float minval = af::min<float>(A);
    float maxval = af::max<float>(A);

    ASSERT_NE(std::isnan(minval), true);
    ASSERT_NE(std::isnan(maxval), true);

    float *h_A = A.host<float>();

    for (int i = 0; i < num; i++) {
        if (!std::isnan(h_A[i])) {
            ASSERT_LE(minval, h_A[i]);
            ASSERT_GE(maxval, h_A[i]);
        }
    }
}

TEST(Count, NaN)
{
    const int num = 10000;
    af::array A = af::round(5 * af::randu(num));
    af::array B = A;

    A(where(A == 2)) = af::NaN;

    ASSERT_EQ(af::count<uint>(A), af::count<uint>(B));
}

TEST(Sum, NaN)
{
    const int num = 10000;
    af::array A = af::randu(num);
    A(where(A < 0.25)) = af::NaN;

    float res = af::sum<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res = af::sum<float>(A, 0);
    float *h_A = A.host<float>();

    float tmp = 0;
    for (int i = 0; i < num; i++) {
        tmp += std::isnan(h_A[i]) ? 0 : h_A[i];
    }

    ASSERT_NEAR(res/num, tmp/num, 1E-5);
}

TEST(Product, NaN)
{
    const int num = 5;
    af::array A = af::randu(num);
    A(2) = af::NaN;

    float res = af::product<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res = af::product<float>(A, 1);
    float *h_A = A.host<float>();

    float tmp = 1;
    for (int i = 0; i < num; i++) {
        tmp *= std::isnan(h_A[i]) ? 1 : h_A[i];
    }

    ASSERT_NEAR(res/num, tmp/num, 1E-5);
}

TEST(AnyAll, NaN)
{
    const int num = 10000;
    af::array A = (af::randu(num) > 0.5).as(f32);
    af::array B = A;

    B(af::where(B == 0)) = af::NaN;

    ASSERT_EQ(af::anyTrue<bool>(B), true);
    ASSERT_EQ(af::allTrue<bool>(B), true);
    ASSERT_EQ(af::anyTrue<bool>(A), true);
    ASSERT_EQ(af::allTrue<bool>(A), false);
}

TEST(MaxAll, IndexedSmall)
{
    const int num = 1000;
    const int st = 10;
    const int en = num - 100;
    af::array a = af::randu(num);
    float b = af::max<float>(a(af::seq(st, en)));

    std::vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}

TEST(MaxAll, IndexedBig)
{
    const int num = 100000;
    const int st = 1000;
    const int en = num - 1000;
    af::array a = af::randu(num);
    float b = af::max<float>(a(af::seq(st, en)));

    std::vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) {
        res = std::max(res, ha[i]);
    }

    ASSERT_EQ(b, res);
}
