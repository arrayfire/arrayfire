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
#include <af/array.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>
#include <af/device.h>
#include "binary_ops.hpp"
#include <utility>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

typedef af_err (*scanFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, scanFunc af_scan>
void scanTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;

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
        ASSERT_EQ(AF_SUCCESS, af_scan(&outArray, inArray, d + off));

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                << " for dim " << d +off
                << std::endl;
        }

        // Delete
        delete[] outData;
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

float randomInterval(float start, float end)
{
    return start + (end - start)*(std::rand()/float(RAND_MAX));
}

int randomInterval(int start, int end)
{
    return start + std::rand()%(end - start);
}

template <typename T>
std::vector<T> createScanKey(af::dim4 dims, int scanDim,
        const std::vector<int> &nodeLengths,
        T keyStart, T keyEnd)
{
    std::srand(0);
    int elemCount = dims.elements();
    std::vector<T> key(elemCount);

    int stride = 1;
    for (int i = 0; i < scanDim; ++i) { stride *= dims[i]; }

    for (int start = 0; start < stride; ++start) {
        T keyval = (T)(0);
        for (int index = start, i = 0;
                index < elemCount;
                index += stride, i = (i+1)%dims[scanDim]) {
            bool isNode = false;
            for (unsigned n = 0; n < nodeLengths.size(); ++n) {
                if (i % nodeLengths[n] == 0) {
                    isNode = true;
                }
            }
            if (isNode && (std::rand()%2)) {
                keyval = randomInterval(keyStart, keyEnd);
            }
            key[index] = keyval;
        }
    }
    return key;
}

template <typename Ti, typename Tk, typename To, af_binary_op op, bool inclusive_scan>
std::pair<std::vector<Ti>, std::vector<To> > createData(af::dim4 dims, const std::vector<Tk> &key,
                int scanDim, Ti dataStart, Ti dataEnd)
{
    std::srand(1);
    Binary<To, op> binOp;
    int elemCount = dims.elements();
    std::vector<To> out(elemCount);
    std::vector<Ti> in(elemCount);

    int stride = 1;
    for (int i = 0; i < scanDim; ++i) { stride *= dims[i]; }

    for (int start = 0; start < stride; ++start) {
        Ti keyval = key[start];
        if (!inclusive_scan) {
            out[start] = binOp.init();
            in[start] = randomInterval(dataStart, dataEnd);
        }
        for (int index = start + (!inclusive_scan)*stride, i = (!inclusive_scan);
                index < elemCount;
                index += stride, i = (i+1)%dims[scanDim]) {
            in[index] = randomInterval(dataStart, dataEnd);
            if ((key[index] != keyval) || (i == 0)) {
                keyval = key[index];
                out[index] = inclusive_scan? (To)in[index] : binOp.init();
            } else {
                To dataval = (To)in[index - (!inclusive_scan)*stride];
                out[index] = binOp(out[index - stride], dataval);
            }
        }
    }
    return std::make_pair(in, out);
}

template<typename Ti, typename To, af_binary_op op, bool inclusive_scan>
void scanByKeyTest(af::dim4 dims, int scanDim, std::vector<int> nodeLengths,
        int keyStart, int keyEnd, Ti dataStart, Ti dataEnd, double eps)
{
    std::vector<int> key = createScanKey<int>(dims, scanDim, nodeLengths, keyStart, keyEnd);
    std::pair<std::vector<Ti>, std::vector<To> > data =
        createData<Ti, int, To, op, inclusive_scan>(dims, key, scanDim, dataStart, dataEnd);
    std::vector<Ti> &in = data.first;
    std::vector<To> &outgold = data.second;
    af::array afkey(dims, key.data());
    af::array afin(dims, in.data());
    af::array afout = af::scanByKey(afkey, afin, scanDim, op, inclusive_scan);

#if 0
    if (scanDim == 1) {
        af_print(afkey.T());
        af_print(afin.T());
        af_print(afout.T());
        af::array afgold(dims, outgold.data());
        af_print(afgold.T());
    }
#endif

    std::vector<To> out(afout.elements());
    afout.host(out.data());
    for(unsigned i = 0; i < out.size(); ++i) {
        ASSERT_NEAR(out[i], outgold[i], eps);
    }
}

#define SCAN_BY_KEY_TEST(FN, X, Y, Z, W, Ti, To, INC, DIM, DSTART, DEND, EPS)   \
TEST(ScanByKey,Test_Scan_By_Key_##FN##_##Ti##_##INC##_##DIM)                    \
{                                                                               \
    af::dim4 dims(X, Y, Z, W);                                                  \
    int scanDim = DIM;                                                          \
    int nodel[] = {37, 256};                                                    \
    std::vector<int> nodeLengths(nodel, nodel+sizeof(nodel)/sizeof(int));       \
    int keyStart = 0;                                                           \
    int keyEnd = 15;                                                            \
    int dataStart = DSTART;                                                     \
    int dataEnd = DEND;                                                         \
    scanByKeyTest<Ti, To, FN, INC>(dims, scanDim, nodeLengths,                  \
            keyStart, keyEnd, dataStart, dataEnd, EPS);                         \
}

SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024+17, 1024, 1, 1,   int,   int,  true, 0,   -15,   15, 1e-5);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024+17, 1024, 1, 1,   int,   int, false, 0,   -15,   15, 1e-5);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024+17, 1024, 1, 1, float, float,  true, 0, -0.25, 0.25, 1e-5);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024+17, 1024, 1, 1, float, float, false, 0, -0.25, 0.25, 1e-5);

TEST(ScanByKey,Test_Scan_By_key_Simple_0)
{
    af::dim4 dims(16, 8, 2, 1);
    int scanDim = 0;
    int nodel[] = {4, 8};
    std::vector<int> nodeLengths(nodel, nodel+sizeof(nodel)/sizeof(int));
    int keyStart = 0;
    int keyEnd = 15;
    int dataStart = 2;
    int dataEnd = 4;
    scanByKeyTest<int, int, AF_BINARY_ADD, false>(dims, scanDim, nodeLengths,
            keyStart, keyEnd, dataStart, dataEnd, 1e-5);
}

TEST(ScanByKey,Test_Scan_By_key_Simple_1)
{
    af::dim4 dims(8, 256+128, 1, 1);
    int scanDim = 1;
    int nodel[] = {4, 8};
    std::vector<int> nodeLengths(nodel, nodel+sizeof(nodel)/sizeof(int));
    int keyStart = 0;
    int keyEnd = 15;
    int dataStart = 2;
    int dataEnd = 4;
    scanByKeyTest<int, int, AF_BINARY_ADD, false>(dims, scanDim, nodeLengths,
            keyStart, keyEnd, dataStart, dataEnd, 1e-5);
}

SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4*1024, 512, 1, 1,   int,   int,  true, 1, -15, 15, 1e-4);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4*1024, 512, 1, 1,   int,   int, false, 1, -15, 15, 1e-4);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4*1024, 512, 1, 1, float, float,  true, 1,  -1,  1, 1e-4);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4*1024, 512, 1, 1, float, float, false, 1,  -1,  1, 1e-4);

#define SCAN_TESTS(FN, TAG, Ti, To)             \
    TEST(Scan,Test_##FN##_##TAG)                \
    {                                           \
        scanTest<Ti, To, af_##FN>(              \
            string(TEST_DIR"/scan/"#FN".test")  \
            );                                  \
    }                                           \

SCAN_TESTS(accum, float   , float     , float     );
SCAN_TESTS(accum, double  , double    , double    );
SCAN_TESTS(accum, int     , int       , int       );
SCAN_TESTS(accum, cfloat  , cfloat    , cfloat    );
SCAN_TESTS(accum, cdouble , cdouble   , cdouble   );
SCAN_TESTS(accum, unsigned, unsigned  , unsigned  );
SCAN_TESTS(accum, intl    , intl      , intl      );
SCAN_TESTS(accum, uintl   , uintl     , uintl     );
SCAN_TESTS(accum, uchar   , uchar     , unsigned  );
SCAN_TESTS(accum, short   , short     , int       );
SCAN_TESTS(accum, ushort  , ushort    , uint      );

TEST(Scan,Test_Scan_Big0)
{
    scanTest<int, int, af_accum>(
        string(TEST_DIR"/scan/big0.test"),
        0
        );
}

TEST(Scan,Test_Scan_Big1)
{
    scanTest<int, int, af_accum>(
        string(TEST_DIR"/scan/big1.test"),
        1
        );
}

///////////////////////////////// CPP ////////////////////////////////////
TEST(Accum, CPP)
{
    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (string(TEST_DIR"/scan/accum.test"),numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<float> in(data[0].begin(), data[0].end());

    if (noDoubleTests<float>()) return;

    af::array input(dims, &(in.front()));

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<float> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        af::array output = af::accum(input, d);

        // Get result
        float *outData;
        outData = new float[dims.elements()];
        output.host((void*)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                << " for dim " << d
                << std::endl;
        }

        // Delete
        delete[] outData;
    }
}
