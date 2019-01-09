/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>

using af::array;
using af::dim4;
using af::dtype_traits;
using af::iota;
using af::topk;
using af::topkFunction;

using std::iota;
using std::make_pair;
using std::min;
using std::mt19937;
using std::ostream;
using std::pair;
using std::random_device;
using std::shuffle;
using std::string;
using std::stringstream;
using std::vector;

template <typename T>
class TopK : public ::testing::Test {};

typedef ::testing::Types<float, double, int, uint> TestTypes;

TYPED_TEST_CASE(TopK, TestTypes);

template <typename T>
void topkTest(const int ndims, const dim_t* dims, const unsigned k,
              const int dim, const af_topk_function order) {
    af_dtype dtype = (af_dtype)dtype_traits<T>::af_type;

    af_array input, output, outindex;

    size_t ielems = 1;
    size_t oelems = 1;

    for (int i = 0; i < ndims; i++) {
        ielems *= dims[i];
        oelems *= (i == dim ? k : dims[i]);
    }

    size_t bCount = ielems / dims[dim];
    size_t bSize  = dims[dim];

    vector<T> inData(ielems);
    iota(begin(inData), end(inData), 0);

    random_device rnd_device;
    mt19937 g(rnd_device());
    shuffle(begin(inData), end(inData), g);

    vector<T> outData(oelems);
    vector<uint> outIdxs(oelems);

    for (size_t b = 0; b < bCount; b++) {
        using KeyValuePair = pair<T, uint>;

        vector<KeyValuePair> kvPairs;
        kvPairs.reserve(((b + 1) * bSize));

        for (size_t i = b * bSize; i < ((b + 1) * bSize); ++i)
            kvPairs.push_back(make_pair(inData[i], (i - b * bSize)));

        if (order == AF_TOPK_MIN) {
            stable_sort(kvPairs.begin(), kvPairs.end(),
                        [](const KeyValuePair& lhs, const KeyValuePair& rhs) {
                            return lhs.first < rhs.first;
                        });
        } else {
            stable_sort(kvPairs.begin(), kvPairs.end(),
                        [](const KeyValuePair& lhs, const KeyValuePair& rhs) {
                            return lhs.first >= rhs.first;
                        });
        }

        auto it = kvPairs.begin();
        for (size_t i = 0; i < k; ++it, ++i) {
            outData[i + b * k] = it->first;
            outIdxs[i + b * k] = it->second;
        }
    }

    ASSERT_SUCCESS(af_create_array(&input, inData.data(), ndims, dims, dtype));
    ASSERT_SUCCESS(af_topk(&output, &outindex, input, k, dim, order));

    vector<T> hovals(oelems);
    vector<uint> hoidxs(oelems);

    ASSERT_SUCCESS(af_get_data_ptr((void*)hovals.data(), output));
    ASSERT_SUCCESS(af_get_data_ptr((void*)hoidxs.data(), outindex));

    for (size_t i = 0; i < oelems; ++i) {
        switch (dtype) {
            case f64:
                EXPECT_DOUBLE_EQ(outData[i], hovals[i]) << "at: " << i;
                break;
            case f32:
                EXPECT_FLOAT_EQ(outData[i], hovals[i]) << "at: " << i;
                break;
            default: EXPECT_EQ(outData[i], hovals[i]); break;
        }
        ASSERT_EQ(outIdxs[i], hoidxs[i]) << "at: " << i;
    }

    ASSERT_SUCCESS(af_release_array(input));
    ASSERT_SUCCESS(af_release_array(output));
    ASSERT_SUCCESS(af_release_array(outindex));
}

TYPED_TEST(TopK, Max1D0) {
    dim_t dims[4] = {100000, 1, 1, 1};
    topkTest<TypeParam>(1, dims, 5, 0, AF_TOPK_MAX);
}

TYPED_TEST(TopK, Max2D0) {
    dim_t dims[4] = {10000, 10, 1, 1};
    topkTest<TypeParam>(2, dims, 3, 0, AF_TOPK_MAX);
}

TYPED_TEST(TopK, Max3D0) {
    dim_t dims[4] = {10000, 10, 10, 1};
    topkTest<TypeParam>(2, dims, 5, 0, AF_TOPK_MAX);
}

TYPED_TEST(TopK, Max4D0) {
    dim_t dims[4] = {10000, 10, 10, 10};
    topkTest<TypeParam>(2, dims, 5, 0, AF_TOPK_MAX);
}

TYPED_TEST(TopK, MIN1D0) {
    dim_t dims[4] = {100000, 1, 1, 1};
    topkTest<TypeParam>(1, dims, 5, 0, AF_TOPK_MIN);
}

TYPED_TEST(TopK, MIN2D0) {
    dim_t dims[4] = {10000, 10, 1, 1};
    topkTest<TypeParam>(2, dims, 3, 0, AF_TOPK_MIN);
}

TYPED_TEST(TopK, MIN3D0) {
    dim_t dims[4] = {10000, 10, 10, 1};
    topkTest<TypeParam>(2, dims, 5, 0, AF_TOPK_MIN);
}

TYPED_TEST(TopK, MIN4D0) {
    dim_t dims[4] = {10000, 10, 10, 10};
    topkTest<TypeParam>(2, dims, 5, 0, AF_TOPK_MIN);
}

TEST(TopK, ValidationCheck_DimN) {
    dim_t dims[4] = {10, 10, 1, 1};
    af_array out, idx, in;
    ASSERT_SUCCESS(af_randu(&in, 2, dims, f32));
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED,
              af_topk(&out, &idx, in, 10, 1, AF_TOPK_MAX));
    ASSERT_SUCCESS(af_release_array(in));
}

TEST(TopK, ValidationCheck_DefaultDim) {
    dim_t dims[4] = {10, 10, 1, 1};
    af_array out, idx, in;
    ASSERT_SUCCESS(af_randu(&in, 4, dims, f32));
    ASSERT_SUCCESS(af_topk(&out, &idx, in, 10, -1, AF_TOPK_MAX));
    ASSERT_SUCCESS(af_release_array(in));
    ASSERT_SUCCESS(af_release_array(out));
    ASSERT_SUCCESS(af_release_array(idx));
}

struct topk_params {
    int d0;
    int d1;
    int k;
    int dim;
    topkFunction order;
};

ostream& operator<<(ostream& os, const topk_params& param) {
    os << "d0: " << param.d0 << " d1: " << param.d1 << " k:  " << param.k
       << " dim: " << param.dim
       << " order: " << ((param.order == AF_TOPK_MAX) ? "MAX" : "MIN");
    return os;
}

class TopKParams : public ::testing::TestWithParam<topk_params> {};

INSTANTIATE_TEST_CASE_P(
    InstantiationName, TopKParams,
    ::testing::Values(topk_params{100, 10, 32, 0, AF_TOPK_MIN},
                      topk_params{100, 10, 64, 0, AF_TOPK_MIN},
                      topk_params{100, 10, 32, 0, AF_TOPK_MAX},
                      topk_params{100, 10, 64, 0, AF_TOPK_MAX},
                      topk_params{100, 10, 5, 0, AF_TOPK_MIN},
                      topk_params{1000, 10, 5, 0, AF_TOPK_MIN},
                      topk_params{10000, 10, 5, 0, AF_TOPK_MIN},
                      topk_params{100, 10, 5, 0, AF_TOPK_MAX},
                      topk_params{1000, 10, 5, 0, AF_TOPK_MAX},
                      topk_params{10000, 10, 5, 0, AF_TOPK_MAX},
                      topk_params{10, 10, 5, 0, AF_TOPK_MIN},
                      topk_params{10, 100, 5, 0, AF_TOPK_MIN},
                      topk_params{10, 1000, 5, 0, AF_TOPK_MIN},
                      topk_params{10, 10000, 5, 0, AF_TOPK_MIN},
                      topk_params{10, 10, 5, 0, AF_TOPK_MAX},
                      topk_params{10, 100, 5, 0, AF_TOPK_MAX},
                      topk_params{10, 1000, 5, 0, AF_TOPK_MAX},
                      topk_params{10, 10000, 5, 0, AF_TOPK_MAX},
                      topk_params{1000, 10, 256, 0, AF_TOPK_MAX}),
    [](const ::testing::TestParamInfo<TopKParams::ParamType> info) {
        stringstream ss;
        ss << "d0_" << info.param.d0 << "_d1_" << info.param.d1 << "_k_"
           << info.param.k << "_dim_" << info.param.dim << "_order_"
           << ((info.param.order == AF_TOPK_MAX) ? string("MAX")
                                                 : string("MIN"));
        return ss.str();
    });

string print_context(int idx0, int idx1, const vector<float>& val,
                     const vector<unsigned>& idx) {
    stringstream ss;
    if (idx0 > 3 && idx1 > 3) {
        for (int i = idx0 - 3; i < idx0 + 3; i++) {
            ss << i << ": " << val[i] << " " << idx[i] << "\n";
        }
    } else {
        int end = min(6, idx0 + 3);
        for (int i = 0; i < end; i++) {
            ss << i << ": " << val[i] << " " << idx[i] << "\n";
        }
    }
    return ss.str();
}

TEST_P(TopKParams, CPP) {
    topk_params params = GetParam();
    int d0             = params.d0;
    int d1             = params.d1;
    int k              = params.k;
    int dim            = params.dim;
    topkFunction order = params.order;

    array in = iota(dim4(d0, d1));

    // reverse the array if the order is ascending
    if (order == AF_TOPK_MIN) { in = -in + (d0 * d1 - 1); }
    array val, idx;
    topk(val, idx, in, k, dim, order);

    vector<float> hval(k * d1);
    vector<unsigned> hidx(k * d1);
    val.host(&hval[0]);
    idx.host(&hidx[0]);

    if (order == AF_TOPK_MIN) {
        for (int j = d1 - 1, i = 0; j > 0; j--) {
            for (int kidx = 0, goldidx = d0 - 1; kidx < k;
                 i++, kidx++, goldidx--) {
                float gold = static_cast<float>(j * d0 + kidx);
                ASSERT_FLOAT_EQ(gold, hval[i])
                    << print_context(i, kidx, hval, hidx);
                ASSERT_EQ(goldidx, hidx[i])
                    << print_context(i, kidx, hval, hidx);
            }
        }
    } else {
        for (int ii = 0, i = 0; ii < d1; ii++) {
            for (int j = d0 - 1; j >= d0 - k; --j, i++) {
                float gold  = static_cast<float>(ii * d0 + j);
                int goldidx = j;
                ASSERT_FLOAT_EQ(gold, hval[i])
                    << print_context(i, 0, hval, hidx);
                ASSERT_EQ(goldidx, hidx[i]) << print_context(i, 0, hval, hidx);
            }
        }
    }
}
