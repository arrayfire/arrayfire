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

template <typename T>
std::vector<T> createScanData(af::dim4 dims, T dataStart, T dataEnd)
{
    int elemCount = dims.elements();
    std::vector<T> in(elemCount);
    for (int i = 0; i < elemCount; ++i) {
        in[i] = randomInterval(dataStart, dataEnd);
    }
    return in;
}

template <typename Ti, typename Tk, typename To, af_binary_op op, bool inclusive_scan>
void verify(af::dim4 dims,
        const std::vector<Ti> &in,
        const std::vector<Tk> &key,
        const std::vector<To> &out,
        int scanDim, double eps)
{
    std::srand(1);
    Binary<To, op> binOp;
    int elemCount = dims.elements();

    int stride = 1;
    for (int i = 0; i < scanDim; ++i) { stride *= dims[i]; }

    for (int start = 0; start < stride; ++start) {
        Tk keyval = key[start];
        To gold = Binary<To, op>::init();
        for (int index = start + (!inclusive_scan)*stride, i = (!inclusive_scan);
                index < elemCount;
                index += stride, i = (i+1)%dims[scanDim]) {
            if ((key[index] != keyval) || (i == 0)) {
                keyval = key[index];
                if (inclusive_scan) {
                    gold = (To)in[index];
                    ASSERT_NEAR(gold, out[index], eps);
                } else {
                    gold = Binary<To, op>::init();
                }
            } else {
                To dataval = (To)in[index - (!inclusive_scan)*stride];
                gold = binOp(gold, dataval);
                ASSERT_NEAR(gold, out[index], eps);
            }
        }
    }
}

template<typename Ti, typename To, af_binary_op op, bool inclusive_scan>
void scanByKeyTest(af::dim4 dims, int scanDim, std::vector<int> nodeLengths,
        int keyStart, int keyEnd, Ti dataStart, Ti dataEnd, double eps)
{
    std::vector<int> key = createScanKey<int>(dims, scanDim, nodeLengths, keyStart, keyEnd);
    std::vector<Ti> in = createScanData<Ti>(dims, dataStart, dataEnd);

    af::array afkey(dims, key.data());
    af::array afin(dims, in.data());
    af::array afout = af::scanByKey(afkey, afin, scanDim, op, inclusive_scan);
    std::vector<To> out(afout.elements());
    afout.host(out.data());

    verify<Ti, int, To, op, inclusive_scan>(dims, in, key, out, scanDim, eps);
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

SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024, 1024, 1, 1,   int,   int,  true, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024, 1024, 1, 1,   int,   int, false, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024, 1024, 1, 1, float, float,  true, 0, -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16*1024, 1024, 1, 1, float, float, false, 0, -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16*1024, 1024, 1, 1,   int,   int,  true, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16*1024, 1024, 1, 1,   int,   int, false, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16*1024, 1024, 1, 1, float, float,  true, 0, -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16*1024, 1024, 1, 1, float, float, false, 0, -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16*1024, 1024, 1, 1,   int,   int,  true, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16*1024, 1024, 1, 1,   int,   int, false, 0,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16*1024, 1024, 1, 1, float, float,  true, 0, -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16*1024, 1024, 1, 1, float, float, false, 0, -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_ADD,  4*1024,  512, 1, 1,   int,   int,  true, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD,  4*1024,  512, 1, 1,   int,   int, false, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD,  4*1024,  512, 1, 1, float, float,  true, 1,   -5,   5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD,  4*1024,  512, 1, 1, float, float, false, 1,   -5,   5, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MIN,  4*1024,  512, 1, 1,   int,   int,  true, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN,  4*1024,  512, 1, 1,   int,   int, false, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN,  4*1024,  512, 1, 1, float, float,  true, 1,   -5,   5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN,  4*1024,  512, 1, 1, float, float, false, 1,   -5,   5, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MAX,  4*1024,  512, 1, 1,   int,   int,  true, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX,  4*1024,  512, 1, 1,   int,   int, false, 1,  -15,  15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX,  4*1024,  512, 1, 1, float, float,  true, 1,   -5,   5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX,  4*1024,  512, 1, 1, float, float, false, 1,   -5,   5, 1e-3);

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
