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
#include <af/traits.hpp>

#include <math.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::freeHost;
using af::tile;
using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Reduce : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, uint, int, intl, uintl,
                         uchar, short, ushort>
    TestTypes;
TYPED_TEST_SUITE(Reduce, TestTypes);

typedef af_err (*reduceFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, reduceFunc af_reduce>
void reduceTest(string pTestFile, int off = 0, bool isSubRef = false,
                const vector<af_seq> seqv = vector<af_seq>()) {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);
    dim4 dims = numDims[0];

    vector<Ti> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(), convert_to<Ti, int>);

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_SUCCESS(
            af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(),
                            (af_dtype)af::dtype_traits<Ti>::af_type));
        ASSERT_SUCCESS(
            af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_SUCCESS(af_release_array(tempArray));
    } else {
        ASSERT_SUCCESS(
            af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(),
                            (af_dtype)af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(af_reduce(&outArray, inArray, d + off));

        af_dtype t;
        af_get_type(&t, outArray);

        // Get result
        vector<To> outData(dims.elements());
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        if (std::equal(currGoldBar.begin(), currGoldBar.end(),
                       outData.begin()) == false) {
            for (size_t elIter = 0; elIter < nElems; ++elIter) {
                EXPECT_EQ(currGoldBar[elIter], outData[elIter])
                    << "at: " << elIter << " for dim " << d + off << endl;
            }
            for (int i = 0; i < (int)nElems; i++) {
                cout << currGoldBar[i] << ", ";
            }

            cout << endl;
            for (int i = 0; i < (int)nElems; i++) {
                cout << outData[i] << ", ";
            }
            FAIL();
        }

        ASSERT_SUCCESS(af_release_array(outArray));
    }

    ASSERT_SUCCESS(af_release_array(inArray));
}

template<typename T, reduceFunc OP>
struct promote_type {
    typedef T type;
};

// char and uchar are promoted to int for sum and product
template<>
struct promote_type<uchar, af_sum> {
    typedef uint type;
};
template<>
struct promote_type<char, af_sum> {
    typedef uint type;
};
template<>
struct promote_type<short, af_sum> {
    typedef int type;
};
template<>
struct promote_type<ushort, af_sum> {
    typedef uint type;
};
template<>
struct promote_type<uchar, af_product> {
    typedef uint type;
};
template<>
struct promote_type<char, af_product> {
    typedef uint type;
};
template<>
struct promote_type<short, af_product> {
    typedef int type;
};
template<>
struct promote_type<ushort, af_product> {
    typedef uint type;
};

#define REDUCE_TESTS(FN)                                                       \
    TYPED_TEST(Reduce, Test_##FN) {                                            \
        reduceTest<TypeParam, typename promote_type<TypeParam, af_##FN>::type, \
                   af_##FN>(string(TEST_DIR "/reduce/" #FN ".test"));          \
    }

REDUCE_TESTS(sum);
REDUCE_TESTS(min);
REDUCE_TESTS(max);

#undef REDUCE_TESTS
#define REDUCE_TESTS(FN, OT)                          \
    TYPED_TEST(Reduce, Test_##FN) {                   \
        reduceTest<TypeParam, OT, af_##FN>(           \
            string(TEST_DIR "/reduce/" #FN ".test")); \
    }

REDUCE_TESTS(any_true, unsigned char);
REDUCE_TESTS(all_true, unsigned char);
REDUCE_TESTS(count, unsigned);

#undef REDUCE_TESTS

TEST(Reduce, Test_Reduce_Big0) {
    reduceTest<int, int, af_sum>(string(TEST_DIR "/reduce/big0.test"), 0);
}

/*
TEST(Reduce,Test_Reduce_Big1)
{
    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big1.test"),
        1
        );
}
*/

/////////////////////////////////// CPP //////////////////////////////////
//
typedef af::array (*ReductionOp)(const af::array &, const int);

using af::allTrue;
using af::anyTrue;
using af::constant;
using af::count;
using af::iota;
using af::max;
using af::min;
using af::NaN;
using af::product;
using af::randu;
using af::round;
using af::seq;
using af::span;
using af::sum;

template<typename Ti, typename To, ReductionOp reduce>
void cppReduceTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);
    dim4 dims = numDims[0];

    vector<Ti> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(), convert_to<Ti, int>);

    array input(dims, &in.front());

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        array output = reduce(input, d);

        // Get result
        vector<To> outData(dims.elements());
        output.host((void *)&outData.front());

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for dim " << d << endl;
        }
    }
}

TEST(Reduce, Test_Sum_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(sum<float>(A, 1), largeDim);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(sum<float>(A, 2), largeDim);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(sum<float>(A, 3), largeDim);
}

TEST(Reduce, Test_Min_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = iota(dim4(1, largeDim, 1, 1));
    ASSERT_EQ(min(A, 1).scalar<float>(), 0.f);
    A = iota(dim4(1, 1, largeDim, 1));
    ASSERT_EQ(min(A, 2).scalar<float>(), 0.f);
    A = iota(dim4(1, 1, 1, largeDim));
    ASSERT_EQ(min(A, 3).scalar<float>(), 0.f);
}

TEST(Reduce, Test_Max_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = iota(dim4(1, largeDim, 1, 1));
    ASSERT_EQ(max(A, 1).scalar<float>(), largeDim - 1);
    A = iota(dim4(1, 1, largeDim, 1));
    ASSERT_EQ(max(A, 2).scalar<float>(), largeDim - 1);
    A = iota(dim4(1, 1, 1, largeDim));
    ASSERT_EQ(max(A, 3).scalar<float>(), largeDim - 1);
}

TEST(Reduce, Test_anyTrue_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(anyTrue(A, 1).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(anyTrue(A, 2).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(anyTrue(A, 3).scalar<char>(), 1);
}

TEST(Reduce, Test_allTrue_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(allTrue(A, 1).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(allTrue(A, 2).scalar<char>(), 1);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(allTrue(A, 3).scalar<char>(), 1);
}

TEST(Reduce, Test_count_Scalar_MaxDim) {
    const size_t largeDim = 65535 * 32 * 8 + 1;
    array A               = constant(1, dim4(1, largeDim, 1, 1));
    ASSERT_EQ(count(A, 1).scalar<unsigned int>(), largeDim);
    A = constant(1, dim4(1, 1, largeDim, 1));
    ASSERT_EQ(count(A, 2).scalar<unsigned int>(), largeDim);
    A = constant(1, dim4(1, 1, 1, largeDim));
    ASSERT_EQ(count(A, 3).scalar<unsigned int>(), largeDim);
}

#define CPP_REDUCE_TESTS(FN, FNAME, Ti, To)                                    \
    TEST(Reduce, Test_##FN##_CPP) {                                            \
        cppReduceTest<Ti, To, FN>(string(TEST_DIR "/reduce/" #FNAME ".test")); \
    }

CPP_REDUCE_TESTS(sum, sum, float, float);
CPP_REDUCE_TESTS(min, min, float, float);
CPP_REDUCE_TESTS(max, max, float, float);
CPP_REDUCE_TESTS(anyTrue, any_true, float, unsigned char);
CPP_REDUCE_TESTS(allTrue, all_true, float, unsigned char);
CPP_REDUCE_TESTS(count, count, float, unsigned);

struct reduce_by_key_params {
    size_t iSize, oSize;
    void *iKeys_;
    void *iVals_;
    void *oKeys_;
    void *oVals_;
    af_dtype kType_, vType_, oType_;
    string testname_;
    virtual ~reduce_by_key_params() {}
};

//
// Reduce By Key tests
//
template<typename Tk, typename Tv, typename To>
struct reduce_by_key_params_t : public reduce_by_key_params {
    vector<Tk> iKeys_;
    vector<Tv> iVals_;
    vector<Tk> oKeys_;
    vector<To> oVals_;
    string testname_;

    reduce_by_key_params_t(vector<Tk> ikeys, vector<Tv> ivals, vector<Tk> okeys,
                           vector<To> ovals, string testname)
        : iKeys_(ikeys)
        , iVals_(ivals)
        , oKeys_(okeys)
        , oVals_(ovals)
        , testname_(testname) {
        reduce_by_key_params::iSize  = iKeys_.size();
        reduce_by_key_params::oSize  = oKeys_.size();
        reduce_by_key_params::iKeys_ = iKeys_.data();
        reduce_by_key_params::iVals_ = iVals_.data();
        reduce_by_key_params::oKeys_ = oKeys_.data();
        reduce_by_key_params::oVals_ = oVals_.data();
        reduce_by_key_params::vType_ = (af_dtype)af::dtype_traits<Tv>::af_type;
        reduce_by_key_params::kType_ = (af_dtype)af::dtype_traits<Tk>::af_type;
        reduce_by_key_params::oType_ = (af_dtype)af::dtype_traits<To>::af_type;
        reduce_by_key_params::testname_ = testname_;
    }
    ~reduce_by_key_params_t() {}
};

array ptrToArray(size_t size, void *ptr, af_dtype type) {
    array res;
    switch (type) {
        case f32: res = array(size, (float *)ptr); break;
        case f64: res = array(size, (double *)ptr); break;
        case c32: res = array(size, (cfloat *)ptr); break;
        case c64: res = array(size, (cdouble *)ptr); break;
        case u32: res = array(size, (unsigned *)ptr); break;
        case s32: res = array(size, (int *)ptr); break;
        case u64: res = array(size, (unsigned long long *)ptr); break;
        case s64: res = array(size, (long long *)ptr); break;
        case u16: res = array(size, (unsigned short *)ptr); break;
        case s16: res = array(size, (short *)ptr); break;
        case b8: res = array(size, (char *)ptr); break;
        case u8: res = array(size, (unsigned char *)ptr); break;
        case f16: res = array(size, (half_float::half *)ptr); break;
    }
    return res;
}

array ptrToArray(af::dim4 size, void *ptr, af_dtype type) {
    array res;
    switch (type) {
        case f32: res = array(size, (float *)ptr); break;
        case f64: res = array(size, (double *)ptr); break;
        case c32: res = array(size, (cfloat *)ptr); break;
        case c64: res = array(size, (cdouble *)ptr); break;
        case u32: res = array(size, (unsigned *)ptr); break;
        case s32: res = array(size, (int *)ptr); break;
        case u64: res = array(size, (unsigned long long *)ptr); break;
        case s64: res = array(size, (long long *)ptr); break;
        case u16: res = array(size, (unsigned short *)ptr); break;
        case s16: res = array(size, (short *)ptr); break;
        case b8: res = array(size, (char *)ptr); break;
        case u8: res = array(size, (unsigned char *)ptr); break;
        case f16: res = array(size, (half_float::half *)ptr); break;
    }
    return res;
}

class ReduceByKeyP : public ::testing::TestWithParam<reduce_by_key_params *> {
   public:
    array keys, vals;
    array keyResGold, valsReducedGold;

    void SetUp() {
        reduce_by_key_params *params = GetParam();
        if (noHalfTests(params->vType_)) {
            GTEST_SKIP() << "Half not supported on this device";
        }
        if (noDoubleTests(GetParam()->vType_)) {
            GTEST_SKIP() << "Double not supported on this device";
        }

        keys = ptrToArray(params->iSize, params->iKeys_, params->kType_);
        vals = ptrToArray(params->iSize, params->iVals_, params->vType_);

        keyResGold = ptrToArray(params->oSize, params->oKeys_, params->kType_);
        valsReducedGold =
            ptrToArray(params->oSize, params->oVals_, params->oType_);
    }

    void TearDown() { delete GetParam(); }
};

template<typename T>
struct generateConsq {
    T vals;

    generateConsq(T v_i = 0) : vals(v_i){};

    T operator()() { return vals++; }
};

template<typename T>
struct generateConst {
    T vals;

    generateConst(T v_i) : vals(v_i){};

    T operator()() { return vals; }
};

template<typename Tk, typename Tv, typename To>
reduce_by_key_params *rbk_unique_data(const string testname, const int testSz,
                                      std::function<Tk()> k_gen,
                                      std::function<Tv()> v_gen) {
    vector<Tk> keys(testSz);
    vector<Tv> vals(testSz);

    generate(begin(keys), end(keys), k_gen);
    generate(begin(vals), end(vals), v_gen);

    vector<Tk> okeys(begin(keys), end(keys));
    auto last = unique(begin(okeys), end(okeys));
    okeys.resize(distance(begin(okeys), last));
    vector<To> ovals(testSz, To(1));
    return new reduce_by_key_params_t<Tk, Tv, To>(keys, vals, okeys, ovals,
                                                  testname);
}

template<typename Tk, typename Tv, typename To>
reduce_by_key_params *rbk_single_data(const string testname, const int testSz,
                                      std::function<Tk()> k_gen,
                                      std::function<Tv()> v_gen) {
    vector<Tk> keys(testSz);
    vector<Tv> vals(testSz);

    generate(begin(keys), end(keys), k_gen);
    generate(begin(vals), end(vals), v_gen);

    vector<Tk> okeys(begin(keys), end(keys));
    auto last = unique(begin(okeys), end(okeys));
    okeys.resize(distance(begin(okeys), last));
    vector<To> ovals(okeys.size(), To(keys.size()));
    return new reduce_by_key_params_t<Tk, Tv, To>(keys, vals, okeys, ovals,
                                                  testname);
}

// clang-format off
template<typename Tk, typename Tv, typename To>
vector<reduce_by_key_params*> genUniqueKeyTests() {
  return {rbk_unique_data<Tk, Tv, To>("unique_key", 31,          generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 32,          generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 33,          generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 127,         generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 128,         generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 129,         generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 1024,        generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 1025,        generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_unique_data<Tk, Tv, To>("unique_key", 1024 * 1025, generateConsq<Tk>(0), generateConst<Tv>(Tv( 1 )))
    };
}

template<typename Tk, typename Tv, typename To>
vector<reduce_by_key_params*> genSingleKeyTests() {
  return {rbk_single_data<Tk, Tv, To>("single_key", 31,         generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 32,         generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 33,         generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 127,        generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 128,        generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 129,        generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 1024,       generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 1025,       generateConst<Tk>(0), generateConst<Tv>(Tv( 1 ))),
          rbk_single_data<Tk, Tv, To>("single_key", 128 * 1025, generateConst<Tk>(0), generateConst<Tv>(Tv( 1 )))
    };
}
// clang-format on

vector<reduce_by_key_params *> generateAllTypes() {
    vector<reduce_by_key_params *> out;
    vector<vector<reduce_by_key_params *>> tmp{
        genUniqueKeyTests<int, float, float>(),
        genSingleKeyTests<int, float, float>(),
        genUniqueKeyTests<unsigned, float, float>(),
        genSingleKeyTests<unsigned, float, float>(),
        genUniqueKeyTests<int, double, double>(),
        genSingleKeyTests<int, double, double>(),
        genUniqueKeyTests<unsigned, double, double>(),
        genSingleKeyTests<unsigned, double, double>(),
        genUniqueKeyTests<int, cfloat, cfloat>(),
        genSingleKeyTests<int, cfloat, cfloat>(),
        genUniqueKeyTests<unsigned, cfloat, cfloat>(),
        genSingleKeyTests<unsigned, cfloat, cfloat>(),
        genUniqueKeyTests<int, cdouble, cdouble>(),
        genSingleKeyTests<int, cdouble, cdouble>(),
        genUniqueKeyTests<unsigned, cdouble, cdouble>(),
        genSingleKeyTests<unsigned, cdouble, cdouble>(),
        genUniqueKeyTests<int, half_float::half, float>(),
        genSingleKeyTests<int, half_float::half, float>(),
        genUniqueKeyTests<unsigned, half_float::half, float>(),
        genSingleKeyTests<unsigned, half_float::half, float>(),
    };

    for (auto &v : tmp) { copy(begin(v), end(v), back_inserter(out)); }
    return out;
}

template<typename TestClass>
string testNameGenerator(
    const ::testing::TestParamInfo<typename TestClass::ParamType> info) {
    af_dtype kt = info.param->kType_;
    af_dtype vt = info.param->vType_;
    size_t size = info.param->iSize;
    std::stringstream s;
    s << info.param->testname_ << "_keyType_" << kt << "_valueType_" << vt
      << "_size_" << size;
    return s.str();
}

INSTANTIATE_TEST_SUITE_P(UniqueKeyTests, ReduceByKeyP,
                         ::testing::ValuesIn(generateAllTypes()),
                         testNameGenerator<ReduceByKeyP>);

TEST_P(ReduceByKeyP, SumDim0) {
    if (noHalfTests(GetParam()->vType_)) {
        GTEST_SKIP() << "Half not supported on this device";
    }
    if (noHalfTests(GetParam()->kType_)) {
        GTEST_SKIP() << "Half not supported on this device";
    }
    if (noDoubleTests(GetParam()->vType_)) {
        GTEST_SKIP() << "Double not supported on this device";
    }
    array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals, 0, 0);

    ASSERT_ARRAYS_EQ(keyResGold, keyRes);
    ASSERT_ARRAYS_NEAR(valsReducedGold, valsReduced, 1e-5);
}

TEST_P(ReduceByKeyP, SumDim2) {
    if (noHalfTests(GetParam()->vType_)) {
        GTEST_SKIP() << "Half not supported on this device";
    }
    if (noHalfTests(GetParam()->kType_)) {
        GTEST_SKIP() << "Half not supported on this device";
    }
    if (noDoubleTests(GetParam()->vType_)) {
        GTEST_SKIP() << "Double not supported on this device";
    }
    const int ntile = 2;
    vals            = tile(vals, 1, ntile, 1, 1);
    vals            = reorder(vals, 1, 2, 0, 3);

    valsReducedGold = tile(valsReducedGold, 1, ntile, 1, 1);
    valsReducedGold = reorder(valsReducedGold, 1, 2, 0, 3);

    array keyRes, valsReduced;
    const int dim       = 2;
    const double nanval = 0.0;
    sumByKey(keyRes, valsReduced, keys, vals, dim, nanval);

    ASSERT_ARRAYS_EQ(keyResGold, keyRes);
    ASSERT_ARRAYS_NEAR(valsReducedGold, valsReduced, 1e-5);
}

TEST(ReduceByKey, MultiBlockReduceSingleval) {
    array keys = constant(0, 1024 * 1024, s32);
    array vals = constant(1, 1024 * 1024, f32);

    array keyResGold      = constant(0, 1);
    array valsReducedGold = constant(1024 * 1024, 1, f32);

    array keyRes, valsReduced;
    sumByKey(keyRes, valsReduced, keys, vals);

    ASSERT_TRUE(allTrue<bool>(keyResGold == keyRes));
    ASSERT_ARRAYS_NEAR(valsReducedGold, valsReduced, 1e-5);
}

void reduce_by_key_test(std::string test_fn) {
    vector<dim4> numDims;
    vector<vector<float>> data;
    vector<vector<float>> tests;
    readTests<float, float, float>(test_fn, numDims, data, tests);

    for (size_t t = 0; t < numDims.size() / 2; ++t) {
        dim4 kdim = numDims[t * 2];
        dim4 vdim = numDims[t * 2 + 1];

        vector<int> in_keys(data[t * 2].begin(), data[t * 2].end());
        vector<float> in_vals(data[t * 2 + 1].begin(), data[t * 2 + 1].end());

        af_array inKeys  = 0;
        af_array inVals  = 0;
        af_array outKeys = 0;
        af_array outVals = 0;
        ASSERT_EQ(
            AF_SUCCESS,
            af_create_array(&inKeys, &in_keys.front(), kdim.ndims(), kdim.get(),
                            (af_dtype)af::dtype_traits<int>::af_type));
        ASSERT_EQ(
            AF_SUCCESS,
            af_create_array(&inVals, &in_vals.front(), vdim.ndims(), vdim.get(),
                            (af_dtype)af::dtype_traits<float>::af_type));

        vector<int> currGoldKeys(tests[t * 2].begin(), tests[t * 2].end());
        vector<float> currGoldVals(tests[t * 2 + 1].begin(),
                                   tests[t * 2 + 1].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS,
                  af_sum_by_key(&outKeys, &outVals, inKeys, inVals, 0));

        dim_t ok0, ok1, ok2, ok3;
        dim_t ov0, ov1, ov2, ov3;
        af_get_dims(&ok0, &ok1, &ok2, &ok3, outKeys);
        af_get_dims(&ov0, &ov1, &ov2, &ov3, outVals);

        // Get result
        vector<int> outKeysVec(ok0 * ok1 * ok2 * ok3);
        vector<float> outValsVec(ov0 * ov1 * ov2 * ov3);

        ASSERT_EQ(AF_SUCCESS,
                  af_get_data_ptr((void *)&outKeysVec.front(), outKeys));
        ASSERT_EQ(AF_SUCCESS,
                  af_get_data_ptr((void *)&outValsVec.front(), outVals));

        size_t nElems = currGoldKeys.size();
        if (std::equal(currGoldKeys.begin(), currGoldKeys.end(),
                       outKeysVec.begin()) == false) {
            for (size_t elIter = 0; elIter < nElems; ++elIter) {
                EXPECT_NEAR(currGoldKeys[elIter], outKeysVec[elIter], 1e-4)
                    << "at: " << elIter << endl;
                EXPECT_NEAR(currGoldVals[elIter], outValsVec[elIter], 1e-4)
                    << "at: " << elIter << endl;
            }
            for (int i = 0; i < (int)nElems; i++) {
                cout << currGoldKeys[i] << ":" << currGoldVals[i] << ", ";
            }

            for (int i = 0; i < (int)nElems; i++) {
                cout << outKeysVec[i] << ":" << outValsVec[i] << ", ";
            }
            FAIL();
        }

        ASSERT_EQ(AF_SUCCESS, af_release_array(outKeys));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outVals));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inKeys));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inVals));
    }
}
TEST(ReduceByKey, MultiBlockReduceContig10) {
    reduce_by_key_test(string(TEST_DIR "/reduce/test_contig10_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceRandom10) {
    reduce_by_key_test(string(TEST_DIR "/reduce/test_random10_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceContig500) {
    reduce_by_key_test(string(TEST_DIR "/reduce/test_contig500_by_key.test"));
}

TEST(ReduceByKey, MultiBlockReduceByKeyRandom500) {
    reduce_by_key_test(string(TEST_DIR "/reduce/test_random500_by_key.test"));
}

TEST(ReduceByKey, productReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    productByKey(reduced_keys, reduced_vals, keys, vals, 0, 1);

    const int goldSz = 5;
    const vector<float> gold_reduce{0, 7, 6, 30, 4};

    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, minReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    minByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const vector<float> gold_reduce{0, 1, 6, 2, 4};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, maxReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    maxByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const vector<float> gold_reduce{0, 7, 6, 5, 4};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, allTrueReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 1, 1, 1, 0, 1, 1, 1};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    allTrueByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const vector<char> gold_reduce{0, 1, 1, 0, 1};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, anyTrueReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 8, 8};
    const float testVals[testSz] = {0, 1, 1, 1, 0, 1, 0, 0};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    anyTrueByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 5;
    const vector<char> gold_reduce{0, 1, 1, 1, 0};

    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, countReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 5};
    const float testVals[testSz] = {0, 1, 1, 1, 0, 1, 1, 1};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    countByKey(reduced_keys, reduced_vals, keys, vals);

    const int goldSz = 4;
    const vector<unsigned> gold_reduce{0, 2, 1, 3};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, ReduceByKeyNans) {
    SKIP_IF_FAST_MATH_ENABLED();
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, NAN, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    array reduced_keys, reduced_vals;
    productByKey(reduced_keys, reduced_vals, keys, vals, 0, 1);

    const int goldSz = 5;
    const vector<float> gold_reduce{0, 7, 6, 30, 4};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, nDim0ReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    const int ntile = 2;
    vals            = tile(vals, af::dim4(1, ntile, ntile, ntile));

    array reduced_keys, reduced_vals;
    const int dim       = 0;
    const double nanval = 0.0;
    sumByKey(reduced_keys, reduced_vals, keys, vals, dim, nanval);

    const dim4 goldSz(5, 2, 2, 2);
    const vector<float> gold_reduce{0, 8, 6, 10, 4, 0, 8, 6, 10, 4,

                                    0, 8, 6, 10, 4, 0, 8, 6, 10, 4,

                                    0, 8, 6, 10, 4, 0, 8, 6, 10, 4,

                                    0, 8, 6, 10, 4, 0, 8, 6, 10, 4};
    ASSERT_VEC_ARRAY_EQ(gold_reduce, goldSz, reduced_vals);
}

TEST(ReduceByKey, nDim1ReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    const int ntile = 2;
    vals            = tile(vals, af::dim4(1, ntile, 1, 1));
    vals            = transpose(vals);

    array reduced_keys, reduced_vals;
    const int dim       = 1;
    const double nanval = 0.0;
    sumByKey(reduced_keys, reduced_vals, keys, vals, dim, nanval);

    const int goldSz                = 5;
    const float gold_reduce[goldSz] = {0, 8, 6, 10, 4};
    vector<float> hreduce(reduced_vals.elements());
    reduced_vals.host(hreduce.data());

    for (int i = 0; i < goldSz * ntile; i++) {
        ASSERT_EQ(gold_reduce[i / ntile], hreduce[i]);
    }
}

TEST(ReduceByKey, nDim2ReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    const int ntile = 2;
    vals            = tile(vals, af::dim4(1, ntile, 1, 1));
    vals            = reorder(vals, 1, 2, 0, 3);

    array reduced_keys, reduced_vals;
    const int dim       = 2;
    const double nanval = 0.0;
    sumByKey(reduced_keys, reduced_vals, keys, vals, dim, nanval);

    const int goldSz                = 5;
    const float gold_reduce[goldSz] = {0, 8, 6, 10, 4};
    vector<float> h_a(reduced_vals.elements());
    reduced_vals.host(h_a.data());

    for (int i = 0; i < goldSz * ntile; i++) {
        ASSERT_EQ(gold_reduce[i / ntile], h_a[i]);
    }
}

TEST(ReduceByKey, nDim3ReduceByKey) {
    const static int testSz      = 8;
    const int testKeys[testSz]   = {0, 2, 2, 9, 5, 5, 5, 8};
    const float testVals[testSz] = {0, 7, 1, 6, 2, 5, 3, 4};

    array keys(testSz, testKeys);
    array vals(testSz, testVals);

    const int ntile = 2;
    vals            = tile(vals, af::dim4(1, ntile, 1, 1));
    vals            = reorder(vals, 1, 2, 3, 0);

    array reduced_keys, reduced_vals;
    const int dim       = 3;
    const double nanval = 0.0;
    sumByKey(reduced_keys, reduced_vals, keys, vals, dim, nanval);

    const int goldSz                = 5;
    const float gold_reduce[goldSz] = {0, 8, 6, 10, 4};
    vector<float> h_a(reduced_vals.elements());
    reduced_vals.host(h_a.data());

    for (int i = 0; i < goldSz * ntile; i++) {
        ASSERT_EQ(gold_reduce[i / ntile], h_a[i]);
    }
}

TEST(Reduce, Test_Product_Global) {
    const int num = 100;
    array a       = 1 + round(5 * randu(num, 1)) / 100;

    float res  = product<float>(a);
    float *h_a = a.host<float>();
    float gold = 1;

    for (int i = 0; i < num; i++) { gold *= h_a[i]; }

    ASSERT_NEAR(gold, res, 1e-3);
    freeHost(h_a);
}

TEST(Reduce, Test_Sum_Global) {
    const int num = 10000;
    array a       = round(2 * randu(num, 1));

    float res  = sum<float>(a);
    float *h_a = a.host<float>();
    float gold = 0;

    for (int i = 0; i < num; i++) { gold += h_a[i]; }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}

TEST(Reduce, Test_Count_Global) {
    const int num = 10000;
    array a       = round(2 * randu(num, 1));
    array b       = a.as(b8);

    int res   = count<int>(b);
    char *h_b = b.host<char>();
    int gold  = 0;

    for (int i = 0; i < num; i++) { gold += h_b[i]; }

    ASSERT_EQ(gold, res);
    freeHost(h_b);
}

TEST(Reduce, Test_min_Global) {
    SUPPORTED_TYPE_CHECK(double);

    const int num = 10000;
    array a       = randu(num, 1, f64);
    double res    = min<double>(a);
    double *h_a   = a.host<double>();
    double gold   = std::numeric_limits<double>::max();

    SUPPORTED_TYPE_CHECK(double);

    for (int i = 0; i < num; i++) { gold = std::min(gold, h_a[i]); }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}

TEST(Reduce, Test_max_Global) {
    const int num = 10000;
    array a       = randu(num, 1);
    float res     = max<float>(a);
    float *h_a    = a.host<float>();
    float gold    = -std::numeric_limits<float>::max();

    for (int i = 0; i < num; i++) { gold = std::max(gold, h_a[i]); }

    ASSERT_EQ(gold, res);
    freeHost(h_a);
}

template<typename T>
void typed_assert_eq(T lhs, T rhs, bool both = true) {
    UNUSED(both);
    ASSERT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<float>(float lhs, float rhs, bool both) {
    UNUSED(both);
    ASSERT_FLOAT_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<double>(double lhs, double rhs, bool both) {
    UNUSED(both);
    ASSERT_DOUBLE_EQ(lhs, rhs);
}

template<>
void typed_assert_eq<cfloat>(cfloat lhs, cfloat rhs, bool both) {
    ASSERT_FLOAT_EQ(real(lhs), real(rhs));
    if (both) { ASSERT_FLOAT_EQ(imag(lhs), imag(rhs)); }
}

template<>
void typed_assert_eq<cdouble>(cdouble lhs, cdouble rhs, bool both) {
    ASSERT_DOUBLE_EQ(real(lhs), real(rhs));
    if (both) { ASSERT_DOUBLE_EQ(imag(lhs), imag(rhs)); }
}

TYPED_TEST(Reduce, Test_All_Global) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    // Input size test
    for (int i = 1; i < 1000; i += 100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam) true);
        array a(2, num / 2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) true, res, false);

        h_vals[3] = false;
        a         = array(2, num / 2, &h_vals.front());

        res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) false, res, false);
    }

    // false value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam) true);
    for (int i = 1; i < 10000; i += 100) {
        h_vals[i] = false;
        array a(2, num / 2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) false, res, false);

        h_vals[i] = true;
    }
}

TYPED_TEST(Reduce, Test_Any_Global) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    // Input size test
    for (int i = 1; i < 1000; i += 100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam) false);
        array a(2, num / 2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) false, res, false);

        h_vals[3] = true;
        a         = array(2, num / 2, &h_vals.front());

        res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) true, res, false);
    }

    // true value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam) false);
    for (int i = 1; i < 10000; i += 100) {
        h_vals[i] = true;
        array a(2, num / 2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        typed_assert_eq((TypeParam) true, res, false);

        h_vals[i] = false;
    }
}

TEST(MinMax, MinMaxNaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num      = 10000;
    array A            = randu(num);
    A(where(A < 0.25)) = NaN;

    float minval = min<float>(A);
    float maxval = max<float>(A);

    ASSERT_NE(std::isnan(minval), true);
    ASSERT_NE(std::isnan(maxval), true);

    float *h_A = A.host<float>();

    for (int i = 0; i < num; i++) {
        if (!std::isnan(h_A[i])) {
            ASSERT_LE(minval, h_A[i]);
            ASSERT_GE(maxval, h_A[i]);
        }
    }

    freeHost(h_A);
}

TEST(MinMax, MinCplxNaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    float real_wnan_data[] = {0.005f, NAN, -6.3f, NAN,      -0.5f,
                              NAN,    NAN, 0.2f,  -1205.4f, 8.9f};

    float imag_wnan_data[] = {NAN,    NAN, -9.0f, -0.005f, -0.3f,
                              0.007f, NAN, 0.1f,  NAN,     4.5f};

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_min_real[] = {-0.5f, 0.2f};
    float gold_min_imag[] = {-0.3f, 0.1f};

    array min_val = af::min(a);

    vector<complex<float>> h_min_val(cols);
    min_val.host(&h_min_val[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_min_val[i].real(), gold_min_real[i]);
        ASSERT_FLOAT_EQ(h_min_val[i].imag(), gold_min_imag[i]);
    }
}

TEST(MinMax, MaxCplxNaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    // 4th element is unusually large to cover the case where
    //  one part holds the largest value among the array,
    //  and the other part is NaN.
    // There's a possibility where the NaN is turned into 0
    //  (since Binary<>::init() will initialize it to 0 in
    //  for complex max op) during the comparisons, and so its
    //  magnitude will determine that that element is the max,
    //  whereas it should have been ignored since its other
    //  part is NaN
    float real_wnan_data[] = {0.005f, NAN, -6.3f, NAN,      -0.5f,
                              NAN,    NAN, 0.2f,  -1205.4f, 8.9f};

    float imag_wnan_data[] = {NAN,    NAN, -9.0f, -0.005f, -0.3f,
                              0.007f, NAN, 0.1f,  NAN,     4.5f};

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_max_real[] = {-6.3f, 8.9f};
    float gold_max_imag[] = {-9.0f, 4.5f};

    array max_val = af::max(a);

    vector<complex<float>> h_max_val(cols);
    max_val.host(&h_max_val[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_max_val[i].real(), gold_max_real[i]);
        ASSERT_FLOAT_EQ(h_max_val[i].imag(), gold_max_imag[i]);
    }
}

TEST(Count, NaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num = 10000;
    array A       = round(5 * randu(num));
    array B       = A;

    A(where(A == 2)) = NaN;

    ASSERT_EQ(count<uint>(A), count<uint>(B));
}

TEST(Sum, NaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num      = 10000;
    array A            = randu(num);
    A(where(A < 0.25)) = NaN;

    float res = sum<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res        = sum<float>(A, 0);
    float *h_A = A.host<float>();

    float tmp = 0;
    for (int i = 0; i < num; i++) { tmp += std::isnan(h_A[i]) ? 0 : h_A[i]; }

    ASSERT_NEAR(res / num, tmp / num, 1E-5);
    freeHost(h_A);
}

TEST(Product, NaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num = 5;
    array A       = randu(num);
    A(2)          = NaN;

    float res = product<float>(A);

    ASSERT_EQ(std::isnan(res), true);

    res        = product<float>(A, 1);
    float *h_A = A.host<float>();

    float tmp = 1;
    for (int i = 0; i < num; i++) { tmp *= std::isnan(h_A[i]) ? 1 : h_A[i]; }

    ASSERT_NEAR(res / num, tmp / num, 1E-5);
    freeHost(h_A);
}

TEST(AnyAll, NaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num = 10000;
    array A       = (randu(num) > 0.5).as(f32);
    array B       = A;

    B(where(B == 0)) = NaN;

    ASSERT_EQ(anyTrue<bool>(B), true);
    ASSERT_EQ(allTrue<bool>(B), true);
    ASSERT_EQ(anyTrue<bool>(A), true);
    ASSERT_EQ(allTrue<bool>(A), false);
}

TEST(MaxAll, IndexedSmall) {
    const int num = 1000;
    const int st  = 10;
    const int en  = num - 100;
    array a       = randu(num);
    float b       = max<float>(a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) { res = std::max(res, ha[i]); }

    ASSERT_EQ(b, res);
}

TEST(MaxAll, IndexedBig) {
    const int num = 100000;
    const int st  = 1000;
    const int en  = num - 1000;
    array a       = randu(num);
    float b       = max<float>(a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) { res = std::max(res, ha[i]); }

    ASSERT_EQ(b, res);
}

TEST(Reduce, KernelName) {
    const int m = 64;
    const int n = 100;
    const int b = 5;

    array in = constant(0, m, n, b);
    for (int i = 0; i < b; i++) {
        array tmp         = randu(m, n);
        in(span, span, i) = tmp;
        ASSERT_EQ(min<float>(in(span, span, i)), min<float>(tmp));
    }
}

TEST(Reduce, AllSmallIndexed) {
    const int len = 512;
    for (int i = 0; i < 1000; ++i) {
        // const int len = 10000;
        array a = af::range(dim4(len, 2));
        array b = a(seq(len / 2), span);
        // af::sync();
        ASSERT_EQ(max<float>(b), len / 2 - 1);
    }
}

TEST(ProductAll, BoolIn_ISSUE2543_All_Ones) {
    ASSERT_EQ(true, product<int>(constant(1, 5, 5, b8)) > 0);
}

TEST(ProductAll, BoolIn_ISSUE2543_Random_Values) {
    array in = randu(5, 5, b8);
    vector<char> hostData(25);
    in.host(hostData.data());
    unsigned int gold = 1;
    for (size_t i = 0; i < hostData.size(); ++i) { gold *= hostData[i]; }
    const unsigned int out = product<unsigned int>(in);
    ASSERT_EQ(gold, out);
}

TEST(Product, BoolIn_ISSUE2543) {
    array A = randu(5, 5, b8);
    ASSERT_ARRAYS_EQ(allTrue(A), product(A));
}

struct reduce_params {
    double element_value;
    dim4 arr_dim;
    dim4 result_dim;
    int reduce_dim;
    reduce_params(double ev, dim4 ad, dim4 result_d, int red_dim)
        : element_value(ev)
        , arr_dim(ad)
        , result_dim(result_d)
        , reduce_dim(red_dim) {}
};

class ReduceHalf : public ::testing::TestWithParam<reduce_params> {};

INSTANTIATE_TEST_SUITE_P(
    SumFirstNonZeroDim, ReduceHalf,
    ::testing::Values(
        reduce_params(1, dim4(10), dim4(1), -1),
        reduce_params(1, dim4(10, 10), dim4(1, 10), -1),
        reduce_params(1, dim4(10, 10, 10), dim4(1, 10, 10), -1),
        reduce_params(1, dim4(10, 10, 10, 10), dim4(1, 10, 10, 10), -1),

        reduce_params(1, dim4(2048), dim4(1), -1),
        reduce_params(1, dim4(2048, 10), dim4(1, 10), -1),
        reduce_params(1, dim4(2048, 10, 10), dim4(1, 10, 10), -1),
        reduce_params(1, dim4(2048, 10, 10, 10), dim4(1, 10, 10, 10), -1),

        reduce_params(1, dim4(2049), dim4(1), -1),
        reduce_params(1, dim4(2049, 10), dim4(1, 10), -1),
        reduce_params(1, dim4(2049, 10, 10), dim4(1, 10, 10), -1),
        reduce_params(1, dim4(2049, 10, 10, 10), dim4(1, 10, 10, 10), -1),

        reduce_params(1, dim4(8192), dim4(1), -1),
        reduce_params(1, dim4(8192, 10), dim4(1, 10), -1),
        reduce_params(1, dim4(8192, 10, 10), dim4(1, 10, 10), -1),
        reduce_params(1, dim4(8192, 10, 10, 10), dim4(1, 10, 10, 10), -1)));

INSTANTIATE_TEST_SUITE_P(
    SumNonZeroDim, ReduceHalf,
    ::testing::Values(
        reduce_params(1.25, dim4(10, 10), dim4(10), 1),
        reduce_params(1.25, dim4(10, 10, 10), dim4(10, 1, 10), 1),
        reduce_params(1.25, dim4(10, 10, 10, 10), dim4(10, 1, 10, 10), 1),

        reduce_params(1.25, dim4(10, 2048), dim4(10), 1),
        reduce_params(1.25, dim4(10, 2048, 10), dim4(10, 1, 10), 1),
        reduce_params(1.25, dim4(10, 2048, 10, 10), dim4(10, 1, 10, 10), 1),

        reduce_params(1.25, dim4(10, 2049), dim4(10), 1),
        reduce_params(1.25, dim4(10, 2049, 10), dim4(10, 1, 10), 1),
        reduce_params(1.25, dim4(10, 2049, 10, 10), dim4(10, 1, 10, 10), 1),

        reduce_params(1.25, dim4(10, 8192), dim4(10), 1),
        reduce_params(1.25, dim4(10, 8192, 10), dim4(10, 1, 10), 1),
        reduce_params(1.25, dim4(10, 8192, 10, 10), dim4(10, 1, 10, 10), 1),

        reduce_params(1.25, dim4(10, 10, 10), dim4(10, 10, 1), 2),
        reduce_params(1.25, dim4(10, 10, 10, 10), dim4(10, 10, 1, 10), 2),

        reduce_params(1.25, dim4(10, 10, 2048), dim4(10, 10, 1), 2),
        reduce_params(1.25, dim4(10, 10, 2048, 10), dim4(10, 10, 1, 10), 2),

        reduce_params(1.25, dim4(10, 10, 2049), dim4(10, 10, 1), 2),
        reduce_params(1.25, dim4(10, 10, 2049, 10), dim4(10, 10, 1, 10), 2),

        reduce_params(1.25, dim4(10, 10, 8192), dim4(10, 10, 1), 2),
        reduce_params(1.25, dim4(10, 10, 8192, 10), dim4(10, 10, 1, 10), 2)));

TEST_P(ReduceHalf, Sum) {
    SUPPORTED_TYPE_CHECK(af_half);
    reduce_params param = GetParam();

    array arr = constant(param.element_value, param.arr_dim, f16);

    size_t elements = 0;
    if (param.reduce_dim == -1) {
        elements = param.arr_dim[0];
    } else {
        elements = param.arr_dim[param.reduce_dim];
    }

    double result_value = param.element_value * elements;
    array gold          = constant(result_value, param.result_dim, f32);

    array result = sum(arr, param.reduce_dim);
    ASSERT_ARRAYS_EQ(gold, result);
}

TEST_P(ReduceHalf, Product) {
    SUPPORTED_TYPE_CHECK(af_half);
    reduce_params param = GetParam();

    array arr = constant(param.element_value, param.arr_dim, f16);

    size_t elements = 0;
    if (param.reduce_dim == -1) {
        elements = param.arr_dim[0];
    } else {
        elements = param.arr_dim[param.reduce_dim];
    }

    float result_value = pow(param.element_value, elements);

    if (std::isinf(result_value)) {
        SUCCEED();
        return;
    }
    array gold = constant(result_value, param.result_dim, f32);

    array result = product(arr, param.reduce_dim);
    ASSERT_ARRAYS_EQ(gold, result);
}

// TODO(umar): HalfMin
TEST(ReduceHalf, Min) {
    SUPPORTED_TYPE_CHECK(af_half);
    float harr[] = {1, 2, 3, 4, 5, 6, 7};
    array arr(7, harr);
    arr       = arr.as(f16);
    array out = min(arr);

    array gold = constant(1, 1, f16);
    ASSERT_ARRAYS_EQ(gold, out);
}

// TODO(umar): HalfMax
TEST(ReduceHalf, Max) {
    SUPPORTED_TYPE_CHECK(af_half);
    float harr[] = {1, 2, 3, 4, 5, 6, 7};
    array arr(7, harr);
    arr       = arr.as(f16);
    array out = max(arr);

    array gold = constant(7, 1, f16);
    ASSERT_ARRAYS_EQ(gold, out);
}

// TODO(umar): HalfCount
TEST(ReduceHalf, Count) {
    SUPPORTED_TYPE_CHECK(af_half);
    float harr[] = {1, 2, 3, 4, 5, 6, 7};
    array arr(7, harr);
    arr       = arr.as(f16);
    array out = count(arr);

    array gold = constant(7, 1, u32);
    ASSERT_ARRAYS_EQ(gold, out);
}

// TODO(umar): HalfAnyTrue
TEST(ReduceHalf, AnyTrue) {
    SUPPORTED_TYPE_CHECK(af_half);
    float harr[] = {1, 2, 3, 4, 5, 6, 7};
    array arr(7, harr);
    arr       = arr.as(f16);
    array out = anyTrue(arr);

    array gold = constant(1, 1, b8);
    ASSERT_ARRAYS_EQ(gold, out);
}

// TODO(umar): HalfAllTrue
TEST(ReduceHalf, AllTrue) {
    SUPPORTED_TYPE_CHECK(af_half);
    float harr[] = {1, 2, 3, 4, 5, 6, 7};
    array arr(7, harr);
    arr       = arr.as(f16);
    array out = allTrue(arr);

    array gold = constant(1, 1, b8);
    ASSERT_ARRAYS_EQ(gold, out);
}

//
// Documentation Snippets

TEST(Reduce, SNIPPET_sum_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    //! [ex_reduce_sum_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 2 3 4 5 6 7 8 9 ];

    array okeys, ovals;
    sumByKey(okeys, ovals, keys, vals);

    // okeys = [ 0  1  0  2 ]
    // ovals = [ 3 12 13 17 ]

    //! [ex_reduce_sum_by_key]

    vector<int> gold_keys   = {0, 1, 0, 2};
    vector<float> gold_vals = {3, 12, 13, 17};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals);
}

TEST(Reduce, SNIPPET_sum_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};

    //! [ex_reduce_sum_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 2 3 4 5  ]
    //         [ 6 7 8 9 10 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    sumByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1  5  9 ],
    //          [ 6 15 19 ]]

    //! [ex_reduce_sum_by_key_dim]

    vector<int> gold_keys   = {1, 0, 2};
    vector<float> gold_vals = {1, 6, 5, 15, 9, 19};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals);
}

TEST(Reduce, SNIPPET_product_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    //! [ex_reduce_product_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 2 3 4 5 6 7 8 9 ];

    array okeys, ovals;
    productByKey(okeys, ovals, keys, vals);

    // okeys = [ 0  1  0  2 ]
    // ovals = [ 2 60 42 72 ]

    //! [ex_reduce_product_by_key]

    vector<int> gold_keys   = {0, 1, 0, 2};
    vector<float> gold_vals = {2, 60, 42, 72};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals);
}

TEST(Reduce, SNIPPET_product_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};

    //! [ex_reduce_product_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 2 3 4 5  ]
    //         [ 6 7 8 9 10 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    productByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1  6 20 ],
    //          [ 6 56 90 ]]

    //! [ex_reduce_product_by_key_dim]

    vector<int> gold_keys   = {1, 0, 2};
    vector<float> gold_vals = {1, 6, 6, 56, 20, 90};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals);
}

TEST(Reduce, SNIPPET_min_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    //! [ex_reduce_min_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 2 3 4 5 6 7 8 9 ];

    array okeys, ovals;
    minByKey(okeys, ovals, keys, vals);

    // okeys = [ 0 1 0 2 ]
    // ovals = [ 1 3 6 8 ]

    //! [ex_reduce_min_by_key]

    vector<int> gold_keys   = {0, 1, 0, 2};
    vector<float> gold_vals = {1, 3, 6, 8};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals);
}

TEST(Reduce, SNIPPET_min_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};

    //! [ex_reduce_min_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 2 3 4 5  ]
    //         [ 6 7 8 9 10 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    minByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1 2 4 ],
    //          [ 6 7 9 ]]

    //! [ex_reduce_min_by_key_dim]

    vector<int> gold_keys   = {1, 0, 2};
    vector<float> gold_vals = {1, 6, 2, 7, 4, 9};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals);
}

TEST(Reduce, SNIPPET_max_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    //! [ex_reduce_max_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 2 3 4 5 6 7 8 9 ];

    array okeys, ovals;
    maxByKey(okeys, ovals, keys, vals);

    // okeys = [ 0 1 0 2 ]
    // ovals = [ 2 5 7 9 ]

    //! [ex_reduce_max_by_key]

    vector<int> gold_keys   = {0, 1, 0, 2};
    vector<float> gold_vals = {2, 5, 7, 9};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals);
}

TEST(Reduce, SNIPPET_max_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};

    //! [ex_reduce_max_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 2 3 4 5  ]
    //         [ 6 7 8 9 10 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    maxByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1  3  5 ],
    //          [ 6  8 10 ]]

    //! [ex_reduce_max_by_key_dim]

    vector<int> gold_keys   = {1, 0, 2};
    vector<float> gold_vals = {1, 6, 3, 8, 5, 10};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals);
}

TEST(Reduce, SNIPPET_alltrue_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 1, 0, 1, 1, 0, 0, 1, 0};

    //! [ex_reduce_alltrue_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 1 0 1 1 0 0 1 0 ];

    array okeys, ovals;
    allTrueByKey(okeys, ovals, keys, vals);

    // okeys = [ 0 1 0 2 ]
    // ovals = [ 1 0 0 0 ]

    //! [ex_reduce_alltrue_by_key]

    vector<int> gold_keys           = {0, 1, 0, 2};
    vector<unsigned char> gold_vals = {1, 0, 0, 0};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals.as(u8));
}

TEST(Reduce, SNIPPET_alltrue_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 0, 1, 1, 1, 0, 0, 1, 1, 1};

    //! [ex_reduce_alltrue_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 1 1 0 1 ]
    //         [ 0 1 0 1 1 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    allTrueByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1 1 0 ],
    //          [ 0 0 1 ]]

    //! [ex_reduce_alltrue_by_key_dim]

    vector<int> gold_keys           = {1, 0, 2};
    vector<unsigned char> gold_vals = {1, 0, 1, 0, 0, 1};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals.as(u8));
}

TEST(Reduce, SNIPPET_anytrue_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 1, 0, 1, 1, 0, 0, 1, 0};

    //! [ex_reduce_anytrue_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 1 0 1 1 0 0 1 0 ];

    array okeys, ovals;
    anyTrueByKey(okeys, ovals, keys, vals);

    // okeys = [ 0 1 0 2 ]
    // ovals = [ 1 0 0 0 ]

    //! [ex_reduce_anytrue_by_key]

    vector<int> gold_keys           = {0, 1, 0, 2};
    vector<unsigned char> gold_vals = {1, 1, 0, 1};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals.as(u8));
}

TEST(Reduce, SNIPPET_anytrue_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 0, 1, 1, 1, 0, 0, 1, 1, 1};

    //! [ex_reduce_anytrue_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 1 1 0 1 ]
    //         [ 0 1 0 1 1 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    anyTrueByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1 1 1 ],
    //          [ 0 1 1 ]]

    //! [ex_reduce_anytrue_by_key_dim]

    vector<int> gold_keys           = {1, 0, 2};
    vector<unsigned char> gold_vals = {1, 0, 1, 1, 1, 1};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals.as(u8));
}

TEST(Reduce, SNIPPET_count_by_key) {
    int hkeys[]   = {0, 0, 1, 1, 1, 0, 0, 2, 2};
    float hvals[] = {1, 1, 0, 1, 1, 0, 0, 1, 0};

    //! [ex_reduce_count_by_key]

    array keys(9, hkeys);  // keys = [ 0 0 1 1 1 0 0 2 2 ]
    array vals(9, hvals);  // vals = [ 1 1 0 1 1 0 0 1 0 ];

    array okeys, ovals;
    countByKey(okeys, ovals, keys, vals);

    // okeys = [ 0 1 0 2 ]
    // ovals = [ 2 2 0 1 ]

    //! [ex_reduce_count_by_key]

    vector<int> gold_keys      = {0, 1, 0, 2};
    vector<unsigned> gold_vals = {2, 2, 0, 1};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(4), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(4), ovals);
}

TEST(Reduce, SNIPPET_count_by_key_dim) {
    int hkeys[] = {1, 0, 0, 2, 2};

    float hvals[] = {1, 0, 1, 1, 1, 0, 0, 1, 1, 1};

    //! [ex_reduce_count_by_key_dim]

    array keys(5, hkeys);
    array vals(2, 5, hvals);

    // keys = [ 1 0 0 2 2 ]

    // vals = [[ 1 1 1 0 1 ]
    //         [ 0 1 0 1 1 ]]

    const int reduce_dim = 1;
    array okeys, ovals;
    countByKey(okeys, ovals, keys, vals, reduce_dim);

    // okeys = [ 1 0 2 ]

    // ovals = [[ 1 2 1 ],
    //          [ 0 1 2 ]]

    //! [ex_reduce_count_by_key_dim]

    vector<int> gold_keys      = {1, 0, 2};
    vector<unsigned> gold_vals = {1, 0, 2, 1, 1, 2};

    ASSERT_VEC_ARRAY_EQ(gold_keys, dim4(3), okeys);
    ASSERT_VEC_ARRAY_EQ(gold_vals, dim4(2, 3), ovals);
}

TEST(RaggedMax, simple) {
    const int testKeys[6]      = {1, 2, 3, 4, 5, 6};
    const unsigned testVals[2] = {9, 2};

    array arr(3, 2, testKeys);
    array keys(1, 2, testVals);

    array ragged_max, idx;
    const int dim = 0;
    max(ragged_max, idx, arr, keys, dim);

    const dim4 goldSz(1, 2);
    const vector<int> gold_reduced{3, 5};
    const vector<unsigned> gold_idx{2, 1};

    ASSERT_VEC_ARRAY_EQ(gold_reduced, goldSz, ragged_max);
    ASSERT_VEC_ARRAY_EQ(gold_idx, goldSz, idx);
}

TEST(RaggedMax, simpleDim1) {
    const int testKeys[8]      = {1, 2, 3, 4, 5, 6, 7, 8};
    const unsigned testVals[2] = {8, 2};

    array arr(2, 4, testKeys);
    array keys(2, 1, testVals);

    array ragged_max, idx;
    const int dim = 1;
    max(ragged_max, idx, arr, keys, dim);

    const dim4 goldSz(2, 1);
    const vector<int> gold_reduced{7, 4};
    const vector<unsigned> gold_idx{3, 1};

    ASSERT_VEC_ARRAY_EQ(gold_reduced, goldSz, ragged_max);
    ASSERT_VEC_ARRAY_EQ(gold_idx, goldSz, idx);
}

struct ragged_params {
    size_t reduceDimLen_;
    int reduceDim_;
    af_dtype lType_, vType_, oType_;
    string testname_;

    virtual ~ragged_params() {}
};

template<typename Tl, typename Tv, typename To>
struct ragged_params_t : public ragged_params {
    string testname_;

    ragged_params_t(size_t reduce_dim_len, int reduce_dim, string testname)
        : testname_(testname) {
        ragged_params::reduceDim_    = reduce_dim;
        ragged_params::reduceDimLen_ = reduce_dim_len;
        ragged_params::lType_        = (af_dtype)af::dtype_traits<Tl>::af_type;
        ragged_params::vType_        = (af_dtype)af::dtype_traits<Tv>::af_type;
        ragged_params::oType_        = (af_dtype)af::dtype_traits<To>::af_type;
        ragged_params::testname_     = testname_;
    }
    ~ragged_params_t() {}
};

class RaggedReduceMaxRangeP : public ::testing::TestWithParam<ragged_params *> {
   public:
    array vals, ragged_lens;
    array valsReducedGold, idxsReducedGold;

    void SetUp() {
        ragged_params *params = GetParam();
        if (noHalfTests(params->vType_)) {
            GTEST_SKIP() << "Half not supported on this device";
        }
        if (noDoubleTests(GetParam()->vType_)) {
            GTEST_SKIP() << "Double not supported on this device";
        }

        const size_t rdim_size = params->reduceDimLen_;
        const int dim          = params->reduceDim_;

        af::dim4 rdim(3, 3, 3, 3);
        rdim[dim] = rdim_size;
        vals      = af::range(rdim, dim, params->vType_);

        rdim[dim]   = 1;
        ragged_lens = af::range(rdim, (dim > 0) ? 0 : 1, params->lType_) + 1;

        valsReducedGold = af::range(rdim, (dim > 0) ? 0 : 1, params->oType_);
        idxsReducedGold = af::range(rdim, (dim > 0) ? 0 : 1, params->lType_);
    }

    void TearDown() { delete GetParam(); }
};

template<typename Tl, typename Tv, typename To>
ragged_params *ragged_range_data(const string testname, const int testSz,
                                 const int rdim) {
    return new ragged_params_t<Tl, Tv, To>(testSz, rdim, testname);
}

// clang-format off
template<typename Tv, typename To>
vector<ragged_params *> genRaggedRangeTests() {
  return {ragged_range_data<unsigned, Tv, To>("ragged_range", 31,          0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 32,          0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 33,          0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 255,         0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 256,         0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 257,         0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024,        0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1025,        0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024 * 1025, 0),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 31,          1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 32,          1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 33,          1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 255,         1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 256,         1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 257,         1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024,        1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1025,        1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024 * 1025, 1),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 31,          2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 32,          2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 33,          2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 255,         2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 256,         2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 257,         2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024,        2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1025,        2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024 * 1025, 2),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 31,          3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 32,          3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 33,          3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 255,         3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 256,         3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 257,         3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024,        3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1025,        3),
          ragged_range_data<unsigned, Tv, To>("ragged_range", 1024 * 1025, 3),
    };
}
// clang-format on

vector<ragged_params *> generateAllTypesRagged() {
    vector<ragged_params *> out;
    vector<vector<ragged_params *>> tmp{
        genRaggedRangeTests<int, int>(), genRaggedRangeTests<float, float>(),
        genRaggedRangeTests<double, double>(),
        genRaggedRangeTests<half_float::half, half_float::half>()};

    for (auto &v : tmp) { copy(begin(v), end(v), back_inserter(out)); }
    return out;
}

template<typename TestClass>
string testNameGeneratorRagged(
    const ::testing::TestParamInfo<typename TestClass::ParamType> info) {
    af_dtype lt = info.param->lType_;
    af_dtype vt = info.param->vType_;
    size_t size = info.param->reduceDimLen_;
    int rdim    = info.param->reduceDim_;
    std::stringstream s;
    s << info.param->testname_ << "_lenType_" << lt << "_valueType_" << vt
      << "_size_" << size << "_reduceDim_" << rdim;
    return s.str();
}

INSTANTIATE_TEST_SUITE_P(RaggedReduceTests, RaggedReduceMaxRangeP,
                         ::testing::ValuesIn(generateAllTypesRagged()),
                         testNameGeneratorRagged<RaggedReduceMaxRangeP>);

TEST_P(RaggedReduceMaxRangeP, rangeMaxTest) {
    if (noHalfTests(GetParam()->vType_)) {
        GTEST_SKIP() << "Half not supported on this device";
    }
    array ragged_max, idx;
    const int dim = GetParam()->reduceDim_;
    max(ragged_max, idx, vals, ragged_lens, dim);

    ASSERT_ARRAYS_EQ(valsReducedGold, ragged_max);
    ASSERT_ARRAYS_EQ(idxsReducedGold, idx);
}

TEST(ReduceByKey, ISSUE_2955) {
    int N                  = 256;
    af::array val          = af::randu(N);
    af::array key          = af::range(af::dim4(N), 0, af::dtype::s32);
    key(seq(127, af::end)) = 1;

    af::array ok, ov;
    af::sumByKey(ok, ov, key, val);
    ASSERT_EQ(ok.dims(0), 128);
    ASSERT_EQ(ov.dims(0), 128);
}

TEST(ReduceByKey, ISSUE_2955_dim) {
    int N                  = 256;
    af::array val          = af::randu(8, N);
    af::array key          = af::range(af::dim4(N), 0, af::dtype::s32);
    key(seq(127, af::end)) = 1;

    af::array ok, ov;
    af::sumByKey(ok, ov, key, val, 1);
    ASSERT_EQ(ok.dims(0), 128);
    ASSERT_EQ(ov.dims(1), 128);
}

TEST(ReduceByKey, ISSUE_3062) {
    size_t N = 129;

    af::array ones  = af::constant(1, N, u32);
    af::array zeros = af::constant(0, N, u32);

    af::array okeys;
    af::array ovalues;

    af::sumByKey(okeys, ovalues, zeros, ones);
    ASSERT_EQ(ovalues.scalar<unsigned>(), 129);

    af::countByKey(okeys, ovalues, zeros, ones);
    ASSERT_EQ(ovalues.scalar<unsigned>(), 129);

    // test reduction on non-zero dimension as well
    ones  = af::constant(1, 2, N, u32);
    zeros = af::constant(0, N, u32);

    af::sumByKey(okeys, ovalues, zeros, ones, 1);
    ASSERT_EQ(ovalues.scalar<unsigned>(), 129);

    af::countByKey(okeys, ovalues, zeros, ones, 1);
    ASSERT_EQ(ovalues.scalar<unsigned>(), 129);
}

TEST(Reduce, Test_Sum_Global_Array) {
    const int num = 513;
    array a       = af::randn(num, 2, 33, 4);

    float res         = af::sum<float>(a);
    array full_reduce = af::sum<af::array>(a);

    float *h_a = a.host<float>();
    float gold = 0.f;

    for (int i = 0; i < a.elements(); i++) { gold += h_a[i]; }

    float max_error =
        std::numeric_limits<float>::epsilon() * (float)a.elements();
    ASSERT_NEAR(gold, res, max_error);
    ASSERT_NEAR(res, full_reduce.scalar<float>(), max_error);
    freeHost(h_a);
}

TEST(Reduce, Test_Product_Global_Array) {
    const int num = 512;
    array a       = 1 + (0.005 * af::randn(num, 2, 3, 4));

    float res         = af::product<float>(a);
    array full_reduce = af::product<af::array>(a);

    float *h_a = a.host<float>();
    float gold = 1.f;

    for (int i = 0; i < a.elements(); i++) { gold *= h_a[i]; }

    float max_error =
        std::numeric_limits<float>::epsilon() * (float)a.elements();
    ASSERT_NEAR(gold, res, max_error);
    ASSERT_NEAR(res, full_reduce.scalar<float>(), max_error);
    freeHost(h_a);
}

TEST(Reduce, Test_Count_Global_Array) {
    const int num = 10000;
    array a       = round(2 * randu(num, 2, 3, 4));
    array b       = a.as(b8);

    int res       = count<int>(b);
    array res_arr = count<af::array>(b);
    char *h_b     = b.host<char>();
    unsigned gold = 0;

    for (int i = 0; i < a.elements(); i++) { gold += h_b[i]; }

    ASSERT_EQ(gold, res);
    ASSERT_EQ(gold, res_arr.scalar<unsigned>());
    freeHost(h_b);
}

TEST(Reduce, Test_min_Global_Array) {
    SUPPORTED_TYPE_CHECK(double);

    const int num = 10000;
    array a       = af::randn(num, 2, 3, 4, f64);
    double res    = min<double>(a);
    array res_arr = min<af::array>(a);
    double *h_a   = a.host<double>();
    double gold   = std::numeric_limits<double>::max();

    SUPPORTED_TYPE_CHECK(double);

    for (int i = 0; i < a.elements(); i++) { gold = std::min(gold, h_a[i]); }

    ASSERT_EQ(gold, res);
    ASSERT_EQ(gold, res_arr.scalar<double>());
    freeHost(h_a);
}

TEST(Reduce, Test_max_Global_Array) {
    const int num = 10000;
    array a       = af::randn(num, 2, 3, 4);
    float res     = max<float>(a);
    array res_arr = max<af::array>(a);
    float *h_a    = a.host<float>();
    float gold    = -std::numeric_limits<float>::max();

    for (int i = 0; i < a.elements(); i++) { gold = std::max(gold, h_a[i]); }

    ASSERT_EQ(gold, res);
    ASSERT_EQ(gold, res_arr.scalar<float>());
    freeHost(h_a);
}

TYPED_TEST(Reduce, Test_All_Global_Array) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    // Input size test
    for (int i = 1; i < 1000; i += 100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam) true);
        array a(2, num / 2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        array res_arr = allTrue<array>(a);
        typed_assert_eq((TypeParam) true, res, false);
        typed_assert_eq((TypeParam) true, (TypeParam)res_arr.scalar<char>(),
                        false);

        h_vals[3] = false;
        a         = array(2, num / 2, &h_vals.front());

        res     = allTrue<TypeParam>(a);
        res_arr = allTrue<array>(a);
        typed_assert_eq((TypeParam) false, res, false);
        typed_assert_eq((TypeParam) false, (TypeParam)res_arr.scalar<char>(),
                        false);
    }

    // false value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam) true);
    for (int i = 1; i < 10000; i += 100) {
        h_vals[i] = false;
        array a(2, num / 2, &h_vals.front());

        TypeParam res = allTrue<TypeParam>(a);
        array res_arr = allTrue<array>(a);
        typed_assert_eq((TypeParam) false, res, false);
        typed_assert_eq((TypeParam) false, (TypeParam)res_arr.scalar<char>(),
                        false);

        h_vals[i] = true;
    }
}

TYPED_TEST(Reduce, Test_Any_Global_Array) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    // Input size test
    for (int i = 1; i < 1000; i += 100) {
        int num = 10 * i;
        vector<TypeParam> h_vals(num, (TypeParam) false);
        array a(2, num / 2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        array res_arr = anyTrue<array>(a);
        typed_assert_eq((TypeParam) false, res, false);
        typed_assert_eq((TypeParam) false, (TypeParam)res_arr.scalar<char>(),
                        false);

        h_vals[3] = true;
        a         = array(2, num / 2, &h_vals.front());

        res     = anyTrue<TypeParam>(a);
        res_arr = anyTrue<array>(a);
        typed_assert_eq((TypeParam) true, (TypeParam)res_arr.scalar<char>(),
                        false);
    }

    // true value location test
    const int num = 10000;
    vector<TypeParam> h_vals(num, (TypeParam) false);
    for (int i = 1; i < 10000; i += 100) {
        h_vals[i] = true;
        array a(2, num / 2, &h_vals.front());

        TypeParam res = anyTrue<TypeParam>(a);
        array res_arr = anyTrue<array>(a);
        typed_assert_eq((TypeParam) true, res, false);
        typed_assert_eq((TypeParam) true, (TypeParam)res_arr.scalar<char>(),
                        false);

        h_vals[i] = false;
    }
}

TEST(Reduce, Test_Sum_Global_Array_nanval) {
    SKIP_IF_FAST_MATH_ENABLED();
    const int num = 100000;
    array a       = af::randn(num, 2, 34, 4);
    a(1, 0, 0, 0) = NAN;
    a(0, 1, 0, 0) = NAN;
    a(0, 0, 1, 0) = NAN;
    a(0, 0, 0, 1) = NAN;

    double nanval     = 0.2;
    float res         = af::sum<float>(a, nanval);
    array full_reduce = af::sum<af::array>(a, nanval);

    float *h_a = a.host<float>();
    float gold = 0.f;

    for (int i = 0; i < a.elements(); i++) {
        gold += (isnan(h_a[i])) ? nanval : h_a[i];
    }
    float max_error =
        std::numeric_limits<float>::epsilon() * (float)a.elements();
    ASSERT_NEAR(gold, res, max_error);
    ASSERT_NEAR(res, full_reduce.scalar<float>(), max_error);
    freeHost(h_a);
}

TEST(Reduce, nanval_issue_3255) {
    SKIP_IF_FAST_MATH_ENABLED();
    SUPPORTED_TYPE_CHECK(double);
    char *info_str;
    af_array ikeys, ivals, okeys, ovals;
    dim_t dims[1] = {8};

    int ikeys_src[8] = {0, 0, 1, 1, 1, 2, 2, 0};
    ASSERT_SUCCESS(af_create_array(&ikeys, ikeys_src, 1, dims, u32));

    int i;
    for (i = 0; i < 8; i++) {
        double ivals_src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        ivals_src[i]        = NAN;
        ASSERT_SUCCESS(af_create_array(&ivals, ivals_src, 1, dims, f64));

        ASSERT_SUCCESS(
            af_product_by_key_nan(&okeys, &ovals, ikeys, ivals, 0, 1.0));
        af::array ovals_cpp(ovals);
        ASSERT_FALSE(af::anyTrue<bool>(af::isNaN(ovals_cpp)));
        ASSERT_SUCCESS(af_release_array(okeys));

        ASSERT_SUCCESS(af_sum_by_key_nan(&okeys, &ovals, ikeys, ivals, 0, 1.0));
        ovals_cpp = af::array(ovals);

        ASSERT_FALSE(af::anyTrue<bool>(af::isNaN(ovals_cpp)));
        ASSERT_SUCCESS(af_release_array(ivals));
        ASSERT_SUCCESS(af_release_array(okeys));
    }
    ASSERT_SUCCESS(af_release_array(ikeys));
}

TEST(Reduce, SNIPPET_algorithm_func_sum) {
    // clang-format off
    //! [ex_algorithm_func_sum]
    //
    // Create a, a 2x3 array
    array a = iota(dim4(2, 3));           // a = [0, 2, 4,
                                          //      1, 3, 5]

    // Create b by summing across the first dimension
    array b = sum(a);        // sum across the first dimension, same as sum(a, 0)

    // Create c by summing across the second dimension
    array c = sum(a, 1);     // sum across the second dimension

    // Create d by summing across the third dimension
    array d = sum(a, 2);     // sum across the third dimension

    // Create e by summing across the fouth dimension
    array e = sum(a, 3);     // sum acorss the fourth dimension

    // Summing across higher dimensions fails due to stepping out of bounds. For example,
    // array f = sum(a0, 4)  // fails due to stepping out of bounds

    //! [ex_algorithm_func_sum]
    // clang-format on

    using std::vector;
    vector<float> gold_a{0, 1, 2, 3, 4, 5};
    vector<float> gold_b{1, 5, 9};
    vector<float> gold_c{6, 9};

    ASSERT_VEC_ARRAY_EQ(gold_a, a.dims(), a);
    ASSERT_VEC_ARRAY_EQ(gold_b, b.dims(), b);
    ASSERT_VEC_ARRAY_EQ(gold_c, c.dims(), c);
    ASSERT_VEC_ARRAY_EQ(gold_a, d.dims(), d);
    ASSERT_VEC_ARRAY_EQ(gold_a, e.dims(), e);
}
