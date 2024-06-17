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
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype_traits;
using af::randu;
using af::range;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class NearestNeighbour : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create lists of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, uchar, short,
                         ushort>
    TestTypes;

template<typename T>
struct otype_t {
    typedef T otype;
};

template<>
struct otype_t<short> {
    typedef int otype;
};

template<>
struct otype_t<ushort> {
    typedef uint otype;
};

template<>
struct otype_t<uchar> {
    typedef uint otype;
};

// register the type list
TYPED_TEST_SUITE(NearestNeighbour, TestTypes);

template<typename T>
void nearestNeighbourTest(string pTestFile, int feat_dim,
                          const af_match_type type) {
    SUPPORTED_TYPE_CHECK(T);

    typedef typename otype_t<T>::otype To;

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<uint>> tests;

    readTests<T, uint, uint>(pTestFile, numDims, in, tests);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];
    af_array query = 0;
    af_array train = 0;
    af_array idx   = 0;
    af_array dist  = 0;

    ASSERT_SUCCESS(af_create_array(&query, &(in[0].front()), qDims.ndims(),
                                   qDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&train, &(in[1].front()), tDims.ndims(),
                                   tDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(
        af_nearest_neighbour(&idx, &dist, query, train, feat_dim, 1, type));

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    To *outDist           = new To[nElems];

    ASSERT_SUCCESS(af_get_data_ptr((void *)outIdx, idx));
    ASSERT_SUCCESS(af_get_data_ptr((void *)outDist, dist));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ((To)goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
    ASSERT_SUCCESS(af_release_array(query));
    ASSERT_SUCCESS(af_release_array(train));
    ASSERT_SUCCESS(af_release_array(idx));
    ASSERT_SUCCESS(af_release_array(dist));
}

/////////////////////////////////////////////////
// SSD
/////////////////////////////////////////////////
TYPED_TEST(NearestNeighbour, NN_SSD_100_1000_Dim0) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/ssd_100_1000_dim0.test"), 0,
        AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_100_1000_Dim1) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/ssd_100_1000_dim1.test"), 1,
        AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_500_5000_Dim0) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/ssd_500_5000_dim0.test"), 0,
        AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_500_5000_Dim1) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/ssd_500_5000_dim1.test"), 1,
        AF_SSD);
}

/////////////////////////////////////////////////
// SAD
/////////////////////////////////////////////////
TYPED_TEST(NearestNeighbour, NN_SAD_100_1000_Dim0) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/sad_100_1000_dim0.test"), 0,
        AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_100_1000_Dim1) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/sad_100_1000_dim1.test"), 1,
        AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_500_5000_Dim0) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/sad_500_5000_dim0.test"), 0,
        AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_500_5000_Dim1) {
    nearestNeighbourTest<TypeParam>(
        string(TEST_DIR "/nearest_neighbour/sad_500_5000_dim1.test"), 1,
        AF_SAD);
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(NearestNeighbourSSD, CPP) {
    vector<dim4> numDims;
    vector<vector<uint>> in;
    vector<vector<uint>> tests;

    readTests<uint, uint, uint>(TEST_DIR
                                "/nearest_neighbour/ssd_500_5000_dim0.test",
                                numDims, in, tests);

    dim4 qDims = numDims[0];
    dim4 tDims = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    try { nearestNeighbour(idx, dist, query, train, 0, 1, AF_SSD); } catch FUNCTION_UNSUPPORTED

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
}

TEST(NearestNeighbourSAD, CPP) {
    vector<dim4> numDims;
    vector<vector<uint>> in;
    vector<vector<uint>> tests;

    readTests<uint, uint, uint>(TEST_DIR
                                "/nearest_neighbour/sad_100_1000_dim1.test",
                                numDims, in, tests);

    dim4 qDims = numDims[0];
    dim4 tDims = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    try { nearestNeighbour(idx, dist, query, train, 1, 1, AF_SAD); } catch FUNCTION_UNSUPPORTED

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
}

TEST(NearestNeighbourSSD, small) {
    const int ntrain            = 1;
    const int nquery            = 5;
    const int nfeat             = 2;
    float train[ntrain * nfeat] = {
        5,
        5,
    };

    float query[5 * nfeat] = {0, 0, 3.5, 4, 5, 5, 6, 5, 8, 6.5};
    array t(nfeat, ntrain, train);
    array q(nfeat, nquery, query);
    array indices;
    array distances;
    try { nearestNeighbour(indices, distances, q, t, 0, 1, AF_SSD); } catch FUNCTION_UNSUPPORTED

    float expectedDistances[nquery] = {
        (5 - 0) * (5 - 0) + (5 - 0) * (5 - 0),
        (5 - 3.5) * (5 - 3.5) + (5 - 4) * (5 - 4),
        (5 - 5) * (5 - 5) + (5 - 5) * (5 - 5),
        (5 - 6) * (5 - 6) + (5 - 5) * (5 - 5),
        (5 - 8) * (5 - 8) + (5 - 6.5) * (5 - 6.5)};

    vector<float> actualDistances(nquery);
    distances.host(&actualDistances[0]);
    for (int i = 0; i < nquery; i++) {
        EXPECT_NEAR(expectedDistances[i], actualDistances[i], 1E-8);
    }
}

TEST(KNearestNeighbourSSD, small) {
    const int ntrain = 5;
    const int nquery = 3;
    const int nfeat  = 2;

    float query[nquery * nfeat] = {5, 5, 0, 0, 10, 10};
    float train[ntrain * nfeat] = {0, 0, 3.5, 4, 5, 5, 6, 5, 8, 6.5};

    array t(nfeat, ntrain, train);
    array q(nfeat, nquery, query);
    array indices;
    array actualDistances;
    const int k = 2;
    try { nearestNeighbour(indices, actualDistances, q, t, 0, k, AF_SSD); } catch FUNCTION_UNSUPPORTED

    vector<float> expectedDistances{
        (5.f - 5.f) * (5.f - 5.f) + (5.f - 5.f) * (5.f - 5.f),
        (5.f - 6.f) * (5.f - 6.f) + (5.f - 5.f) * (5.f - 5.f),

        (0.f - 0.f) * (0.f - 0.f) + (0.f - 0.f) * (0.f - 0.f),
        (0.f - 3.5f) * (0.f - 3.5f) + (0.f - 4.f) * (0.f - 4.f),

        (10.f - 8.f) * (10.f - 8.f) + (10.f - 6.5f) * (10.f - 6.5f),
        (10.f - 6.f) * (10.f - 6.f) + (10.f - 5.f) * (10.f - 5.f)};

    ASSERT_VEC_ARRAY_NEAR(expectedDistances, dim4(nfeat, nquery),
                          actualDistances, 1E-8);
}

struct nearest_neighbors_params {
    string testname_;
    int k_, nfeat_, ntrain_, nquery_;
    int feat_dim_;
    dim4 qdims_, tdims_, idims_, ddims_;
    vector<float> query_;
    vector<float> train_;
    vector<unsigned int> indices_;
    vector<float> dists_;

    nearest_neighbors_params(string testname, int k, int feat_dim, array query,
                             array train, array indices, array dists)
        : testname_(testname)
        , k_(k)
        , feat_dim_(feat_dim)
        , query_(query.elements())
        , train_(train.elements())
        , indices_(indices.elements())
        , dists_(dists.elements()) {
        qdims_ = query.dims();
        tdims_ = train.dims();
        idims_ = indices.dims();
        ddims_ = dists.dims();

        query.host(query_.data());
        train.host(train_.data());
        indices.host(indices_.data());
        dists.host(dists_.data());
    }
};

template<typename TestClass>
string testNameGenerator(
    const ::testing::TestParamInfo<typename TestClass::ParamType> info) {
    return info.param.testname_;
}

class NearestNeighborsTest
    : public ::testing::TestWithParam<nearest_neighbors_params> {};
class KNearestNeighborsTest
    : public ::testing::TestWithParam<nearest_neighbors_params> {};

nearest_neighbors_params single_knn_data(const string testname,
                                         const int nquery, const int ntrain,
                                         const int nfeat, const int k,
                                         const int feat_dim) {
    array indices, dists;
    array query, train;
    if (feat_dim == 0) {
        query = constant(0, nfeat, nquery);
        train = constant(1, nfeat, ntrain);
    } else {
        query = constant(0, nquery, nfeat);
        train = constant(1, ntrain, nfeat);
    }

    indices = constant(0, k, nquery, u32);
    dists   = constant(nfeat, k, nquery);

    return nearest_neighbors_params(testname, k, feat_dim, query, train,
                                    indices, dists);
}

nearest_neighbors_params knn_data(const string testname, const int nquery,
                                  const int ntrain, const int nfeat,
                                  const int k, const int feat_dim) {
    array indices, dists;
    array query, train;
    if (feat_dim == 0) {
        query = constant(0, nfeat, nquery);
        train = range(dim4(nfeat, ntrain), 1);
    } else {
        query = constant(0, nquery, nfeat);
        train = range(dim4(ntrain, nfeat), 0);
    }

    indices = range(dim4(k, nquery), 0, u32);
    dists   = range(dim4(k, nquery));
    dists *= dists;

    return nearest_neighbors_params(testname, k, feat_dim, query, train,
                                    indices, dists);
}

vector<nearest_neighbors_params> genNNTests() {
    return {
        single_knn_data("1q1t", 1, 1, 10, 1, 0),
        single_knn_data("1q10t", 1, 10, 10, 1, 0),
        single_knn_data("1q100t", 1, 100, 10, 1, 0),
        single_knn_data("1q1000t", 1, 1000, 10, 1, 0),
        single_knn_data("1q100000t", 1, 10000, 10, 1, 0),
        single_knn_data("10q1t", 10, 1, 10, 1, 0),
        single_knn_data("100q1t", 100, 1, 10, 1, 0),
        single_knn_data("1000q1t", 1000, 1, 10, 1, 0),
        single_knn_data("10000q1t", 10000, 1, 10, 1, 0),
        single_knn_data("100000q1t", 10000, 1, 10, 1, 0),
        single_knn_data("1q1tfl1", 10, 1, 1, 1, 0),
        single_knn_data("1q1tfl2", 10, 1, 2, 1, 0),
        single_knn_data("1q1tfl4", 10, 1, 4, 1, 0),
        single_knn_data("1q1tfl8", 10, 1, 8, 1, 0),
        single_knn_data("1q1tfl16", 10, 1, 16, 1, 0),
        single_knn_data("1q1tfl32", 10, 1, 32, 1, 0),
        single_knn_data("1q1tfl64", 10, 1, 64, 1, 0),
        single_knn_data("1q1tfl128", 10, 1, 128, 1, 0),
        single_knn_data("1q1tfl256", 10, 1, 256, 1, 0),
        single_knn_data("1q1tfl10000", 10, 1, 10000, 1, 0),
        single_knn_data("10q1t1d", 10, 1, 10, 1, 1),
        single_knn_data("100q1t1d", 100, 1, 10, 1, 1),
        single_knn_data("1000q1t1d", 1000, 1, 10, 1, 1),
        single_knn_data("10000q1t1d", 10000, 1, 10, 1, 1),
        single_knn_data("100000q1t1d", 10000, 1, 10, 1, 1),
    };
}

vector<nearest_neighbors_params> genKNNTests() {
    return {knn_data("1q1000t1k", 1, 1000, 1, 1, 0),
            knn_data("1q1000t2k", 1, 1000, 1, 2, 0),
            knn_data("1q1000t4k", 1, 1000, 1, 4, 0),
            knn_data("1q1000t8k", 1, 1000, 1, 8, 0),
            knn_data("1q1000t16k", 1, 1000, 1, 16, 0),
            knn_data("1q1000t32k", 1, 1000, 1, 32, 0),
            knn_data("1q1000t64k", 1, 1000, 1, 64, 0),
            knn_data("1q1000t128k", 1, 1000, 1, 128, 0),
            knn_data("1q1000t256k", 1, 1000, 1, 256, 0)};
}

INSTANTIATE_TEST_SUITE_P(KNearestNeighborsSSD, NearestNeighborsTest,
                         ::testing::ValuesIn(genNNTests()),
                         testNameGenerator<NearestNeighborsTest>);

INSTANTIATE_TEST_SUITE_P(KNearestNeighborsSSD, KNearestNeighborsTest,
                         ::testing::ValuesIn(genKNNTests()),
                         testNameGenerator<KNearestNeighborsTest>);

TEST_P(NearestNeighborsTest, SingleQTests) {
    nearest_neighbors_params params = GetParam();
    array query = array(params.qdims_, params.query_.data());
    array train = array(params.tdims_, params.train_.data());

    const int k        = params.k_;
    const int feat_dim = params.feat_dim_;

    array indices, distances;

    try { nearestNeighbour(indices, distances, query, train, feat_dim, k, AF_SSD);
    } catch FUNCTION_UNSUPPORTED

    array indices_gold(params.idims_, params.indices_.data());
    array distances_gold(params.ddims_, params.dists_.data());

    ASSERT_ARRAYS_EQ(indices_gold, indices);
    ASSERT_ARRAYS_NEAR(distances_gold, distances, 1e-5);
}

TEST_P(KNearestNeighborsTest, SingleQTests) {
    nearest_neighbors_params params = GetParam();

    array query = array(params.qdims_, params.query_.data());
    array train = array(params.tdims_, params.train_.data());

    const int k        = params.k_;
    const int feat_dim = params.feat_dim_;

    array indices, distances;

    try { nearestNeighbour(indices, distances, query, train, feat_dim, k, AF_SSD);
    } catch FUNCTION_UNSUPPORTED

    array indices_gold(params.idims_, params.indices_.data());
    array distances_gold(params.ddims_, params.dists_.data());

    ASSERT_ARRAYS_EQ(indices_gold, indices);
    ASSERT_ARRAYS_NEAR(distances_gold, distances, 1e-5);
}

TEST(KNearestNeighbours, InvalidNegativeK) {
    const int ntrain = 500;
    const int nquery = 1;
    const int nfeat  = 2;

    array t = randu(nfeat, ntrain);
    array q = randu(nfeat, nquery);

    array indices;
    array distances;
    int k = -1;
    ASSERT_THROW(nearestNeighbour(indices, distances, q, t, 0, k, AF_SSD),
                 af::exception);
}

TEST(KNearestNeighbours, InvalidLargeK) {
    const int ntrain = 500;
    const int nquery = 1;
    const int nfeat  = 2;

    array t = randu(nfeat, ntrain);
    array q = randu(nfeat, nquery);

    array indices;
    array distances;
    int k = 257;
    ASSERT_THROW(nearestNeighbour(indices, distances, q, t, 0, k, AF_SSD),
                 af::exception);
}

TEST(NearestNeighbour, DocSnippet1) {
    //! [ex_nearest_1]
    float h_pts[6] = {1.f, 2.f, 3.f, 8.f, 9.f, 10.f};
    array pts(dim4(1, 6), h_pts);
    //  1.   2.   3.   8.   9.   10.

    float h_query = 1.25f;
    array query(dim4(1), &h_query);
    //  1.25

    array idx;
    array dist;
    try { nearestNeighbour(idx, dist, query, pts, 0, 3); } catch FUNCTION_UNSUPPORTED
    // idx
    //  0.
    //  1.
    //  2.
    //
    // dist
    //  0.0625
    //  0.5625
    //  3.0625

    //! [ex_nearest_1]

    unsigned int h_gold_idx[3] = {0, 1, 2};
    float h_gold_dist[3]       = {0.0625f, 0.5625f, 3.0625f};
    array gold_idx(dim4(3), h_gold_idx);
    array gold_dist(dim4(3), h_gold_dist);
    ASSERT_ARRAYS_EQ(gold_idx, idx);
    ASSERT_ARRAYS_EQ(gold_dist, dist);
}

TEST(NearestNeighbour, DocSnippet2) {
    //! [ex_nearest_2]
    float h_pts[18] = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
                       8.f, 9.f, 1.f, 9.f, 8.f, 1.f, 9.f, 9.f, 1.f};
    array pts(dim4(3, 6), h_pts);
    //  0.    1.    0.    8.    9.    9.
    //  0.    0.    1.    9.    8.    9.
    //  0.    0.    0.    1.    1.    1.

    float h_query[6] = {1.5f, 0.f, 0.f, 7.5f, 9.f, 1.f};
    array query(dim4(3, 2), h_query);
    //  1.5   7.5
    //  0.    9.
    //  0.    1.

    array idx;
    array dist;
    try { nearestNeighbour(idx, dist, query, pts, 0, 3); } catch FUNCTION_UNSUPPORTED
    // idx
    //  1     3
    //  0     5
    //  2     4
    //
    // dist
    //  0.25  0.25
    //  2.25  2.25
    //  3.25  3.25
    //! [ex_nearest_2]

    unsigned int h_gold_idx[6] = {1, 0, 2, 3, 5, 4};
    float h_gold_dist[6]       = {0.25f, 2.25f, 3.25f, 0.25f, 2.25f, 3.25f};
    array gold_idx(dim4(3, 2), h_gold_idx);
    array gold_dist(dim4(3, 2), h_gold_dist);
    ASSERT_ARRAYS_EQ(gold_idx, idx);
    ASSERT_ARRAYS_EQ(gold_dist, dist);
}
