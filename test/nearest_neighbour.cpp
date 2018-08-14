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
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::endl;
using std::vector;
using std::string;
using af::array;
using af::cfloat;
using af::cdouble;
using af::dim4;
using af::dtype_traits;

template<typename T>
class NearestNeighbour : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create lists of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, uchar, short, ushort> TestTypes;

template<typename T>
struct otype_t
{
    typedef T otype;
};

template<>
struct otype_t<short>
{
    typedef int otype;
};

template<>
struct otype_t<ushort>
{
    typedef uint otype;
};

template<>
struct otype_t<uchar>
{
    typedef uint otype;
};

// register the type list
TYPED_TEST_CASE(NearestNeighbour,  TestTypes);

template<typename T>
void nearestNeighbourTest(string pTestFile, int feat_dim, const af_match_type type)
{
    if (noDoubleTests<T>()) return;

    typedef typename otype_t<T>::otype To;

    vector<dim4>         numDims;
    vector<vector<T> >   in;
    vector<vector<uint> >  tests;

    readTests<T, uint, uint>(pTestFile, numDims, in, tests);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];
    af_array query = 0;
    af_array train = 0;
    af_array idx   = 0;
    af_array dist  = 0;

    ASSERT_SUCCESS(af_create_array(&query, &(in[0].front()),
                qDims.ndims(), qDims.get(), (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&train, &(in[1].front()),
                tDims.ndims(), tDims.get(), (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_nearest_neighbour(&idx, &dist, query, train, feat_dim, 1, type));

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    To *outDist           = new To[nElems];

    ASSERT_SUCCESS(af_get_data_ptr((void*)outIdx,  idx));
    ASSERT_SUCCESS(af_get_data_ptr((void*)outDist, dist));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ((To)goldDist[elIter], outDist[elIter])<< "at: " << elIter<< endl;
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
TYPED_TEST(NearestNeighbour, NN_SSD_100_1000_Dim0)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/ssd_100_1000_dim0.test"), 0, AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_100_1000_Dim1)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/ssd_100_1000_dim1.test"), 1, AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_500_5000_Dim0)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/ssd_500_5000_dim0.test"), 0, AF_SSD);
}

TYPED_TEST(NearestNeighbour, NN_SSD_500_5000_Dim1)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/ssd_500_5000_dim1.test"), 1, AF_SSD);
}

/////////////////////////////////////////////////
// SAD
/////////////////////////////////////////////////
TYPED_TEST(NearestNeighbour, NN_SAD_100_1000_Dim0)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/sad_100_1000_dim0.test"), 0, AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_100_1000_Dim1)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/sad_100_1000_dim1.test"), 1, AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_500_5000_Dim0)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/sad_500_5000_dim0.test"), 0, AF_SAD);
}

TYPED_TEST(NearestNeighbour, NN_SAD_500_5000_Dim1)
{
    nearestNeighbourTest<TypeParam>(string(TEST_DIR"/nearest_neighbour/sad_500_5000_dim1.test"), 1, AF_SAD);
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(NearestNeighbourSSD, CPP)
{
    vector<dim4>         numDims;
    vector<vector<uint> >     in;
    vector<vector<uint> >  tests;

    readTests<uint, uint, uint>(TEST_DIR"/nearest_neighbour/ssd_500_5000_dim0.test", numDims, in, tests);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    nearestNeighbour(idx, dist, query, train, 0, 1, AF_SSD);

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])<< "at: " << elIter<< endl;
    }

    delete[] outIdx;
    delete[] outDist;
}

TEST(NearestNeighbourSAD, CPP)
{
    vector<dim4>         numDims;
    vector<vector<uint> >     in;
    vector<vector<uint> >  tests;

    readTests<uint, uint, uint>(TEST_DIR"/nearest_neighbour/sad_100_1000_dim1.test", numDims, in, tests);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    nearestNeighbour(idx, dist, query, train, 1, 1, AF_SAD);

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])<< "at: " << elIter<< endl;
    }

    delete[] outIdx;
    delete[] outDist;
}

TEST(NearestNeighbourSSD, small)
{
    const int ntrain = 1;
    const int nquery = 5;
    const int nfeat  = 2;
    float train[ntrain * nfeat] = {
        5, 5,
    };

    float query[5 * nfeat] = {
        0, 0,
        3.5, 4,
        5, 5,
        6, 5,
        8, 6.5
    };
    array t(nfeat, ntrain, train);
    array q(nfeat, nquery, query);
    array indices;
    array distances;
    nearestNeighbour(indices, distances, q, t, 0, 1, AF_SSD);

    float expectedDistances[nquery] = {
        (5 - 0) * (5 - 0) + (5 - 0) * (5 - 0),
        (5 - 3.5) * (5 - 3.5) + (5 - 4) * (5 - 4),
        (5 - 5) * (5 - 5) + (5 - 5) * (5 - 5),
        (5 - 6) * (5 - 6) + (5 - 5) * (5 - 5),
        (5 - 8) * (5 - 8) + (5 - 6.5) * (5 - 6.5)
    };

    vector<float> actualDistances(nquery);
    distances.host(&actualDistances[0]);
    for (int i = 0; i < nquery; i++)
    {
        EXPECT_NEAR(expectedDistances[i], actualDistances[i], 1E-8);
    }
}

TEST(KNearestNeighbourSSD, small)
{
    const int ntrain = 5;
    const int nquery = 3;
    const int nfeat  = 2;

    float query[nquery * nfeat] = {
        5, 5,
        0, 0,
       10, 10,
    };

    float train[ntrain * nfeat] = {
        0, 0,
        3.5, 4,
        5, 5,
        6, 5,
        8, 6.5
    };

    array t(nfeat, ntrain, train);
    array q(nfeat, nquery, query);
    array indices;
    array distances;
    const int k = 2;
    nearestNeighbour(indices, distances, q, t, 0, k, AF_SSD);

    float expectedDistances[nquery * ntrain] = {
        (5 - 5) * (5 - 5) + (5 - 5) * (5 - 5),
        (5 - 6) * (5 - 6) + (5 - 5) * (5 - 5),

        (0 - 0)   * (0 - 0) + (0 - 0)   * (0 - 0),
        (0 - 3.5) * (0 - 4) + (0 - 3.5) * (0 - 4),

        (10 - 8)   * (10 - 8)   + (10 - 6.5) * (10 - 6.5),
        (10 - 6)   * (10 - 5)   + (10 - 6)   * (10 - 5)
    };

    vector<float> actualDistances(nquery);
    distances.host(&actualDistances[0]);
    for (int i = 0; i < nquery; i++)
    {
        EXPECT_NEAR(expectedDistances[i], actualDistances[i], 1E-8);
    }
}
