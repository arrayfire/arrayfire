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
#include <af/array.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "binary_ops.hpp"

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using std::cout;
using std::endl;
using std::string;
using std::vector;

float randomInterval(float start, float end) {
    return start + (end - start) * (std::rand() / float(RAND_MAX));
}

int randomInterval(int start, int end) {
    return start + std::rand() % (end - start);
}

template<typename T>
vector<T> createScanKey(dim4 dims, int scanDim, const vector<int> &nodeLengths,
                        T keyStart, T keyEnd) {
    std::srand(0);
    int elemCount = dims.elements();
    vector<T> key(elemCount);

    int stride = 1;
    for (int i = 0; i < scanDim; ++i) { stride *= dims[i]; }

    for (int start = 0; start < stride; ++start) {
        T keyval = (T)(0);
        for (int index = start, i = 0; index < elemCount;
             index += stride, i   = (i + 1) % dims[scanDim]) {
            bool isNode = false;
            for (unsigned n = 0; n < nodeLengths.size(); ++n) {
                if (i % nodeLengths[n] == 0) { isNode = true; }
            }
            if (isNode && (std::rand() % 2)) {
                keyval = randomInterval(keyStart, keyEnd);
            }
            key[index] = keyval;
        }
    }
    return key;
}

template<typename T>
vector<T> createScanData(dim4 dims, T dataStart, T dataEnd) {
    int elemCount = dims.elements();
    vector<T> in(elemCount);
    for (int i = 0; i < elemCount; ++i) {
        in[i] = randomInterval(dataStart, dataEnd);
    }
    return in;
}

template<typename Ti, typename Tk, typename To, af_binary_op op,
         bool inclusive_scan>
void verify(dim4 dims, const vector<Ti> &in, const vector<Tk> &key,
            const vector<To> &out, int scanDim, double eps) {
    std::srand(1);
    Binary<To, op> binOp;
    int elemCount = dims.elements();

    int stride = 1;
    for (int i = 0; i < scanDim; ++i) { stride *= dims[i]; }

    for (int start = 0; start < stride; ++start) {
        Tk keyval = key[start];
        To gold   = binOp.init();
        for (int index = start + (!inclusive_scan) * stride,
                 i     = (!inclusive_scan);
             index < elemCount; index += stride, i = (i + 1) % dims[scanDim]) {
            if ((key[index] != keyval) || (i == 0)) {
                keyval = key[index];
                if (inclusive_scan) {
                    gold = (To)in[index];
                    ASSERT_NEAR(gold, out[index], eps);
                } else {
                    gold = binOp.init();
                }
            } else {
                To dataval = (To)in[index - (!inclusive_scan) * stride];
                gold       = binOp(gold, dataval);
                ASSERT_NEAR(gold, out[index], eps);
            }
        }
    }
}

template<typename Ti, typename To, af_binary_op op, bool inclusive_scan>
void scanByKeyTest(dim4 dims, int scanDim, vector<int> nodeLengths,
                   int keyStart, int keyEnd, Ti dataStart, Ti dataEnd,
                   double eps) {
    vector<int> key =
        createScanKey<int>(dims, scanDim, nodeLengths, keyStart, keyEnd);
    vector<Ti> in = createScanData<Ti>(dims, dataStart, dataEnd);

    array afkey(dims, key.data());
    array afin(dims, in.data());
    array afout;
    try { afout = scanByKey(afkey, afin, scanDim, op, inclusive_scan);
    } catch FUNCTION_UNSUPPORTED
    vector<To> out(afout.elements());
    afout.host(out.data());

    verify<Ti, int, To, op, inclusive_scan>(dims, in, key, out, scanDim, eps);
}

#define SCAN_BY_KEY_TEST(FN, X, Y, Z, W, Ti, To, INC, DIM, DSTART, DEND, EPS) \
    TEST(ScanByKey, Test_Scan_By_Key_##FN##_##Ti##_##INC##_##DIM) {           \
        dim4 dims(X, Y, Z, W);                                                \
        int scanDim = DIM;                                                    \
        int nodel[] = {37, 256};                                              \
        vector<int> nodeLengths(nodel, nodel + sizeof(nodel) / sizeof(int));  \
        int keyStart  = 0;                                                    \
        int keyEnd    = 15;                                                   \
        int dataStart = DSTART;                                               \
        int dataEnd   = DEND;                                                 \
        scanByKeyTest<Ti, To, FN, INC>(dims, scanDim, nodeLengths, keyStart,  \
                                       keyEnd, dataStart, dataEnd, EPS);      \
    }

SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16 * 1024, 1024, 1, 1, int, int, true, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16 * 1024, 1024, 1, 1, int, int, false, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16 * 1024, 1024, 1, 1, float, float, true, 0,
                 -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 16 * 1024, 1024, 1, 1, float, float, false, 0,
                 -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16 * 1024, 1024, 1, 1, int, int, true, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16 * 1024, 1024, 1, 1, int, int, false, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16 * 1024, 1024, 1, 1, float, float, true, 0,
                 -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 16 * 1024, 1024, 1, 1, float, float, false, 0,
                 -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16 * 1024, 1024, 1, 1, int, int, true, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16 * 1024, 1024, 1, 1, int, int, false, 0, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16 * 1024, 1024, 1, 1, float, float, true, 0,
                 -5.0, 5.0, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 16 * 1024, 1024, 1, 1, float, float, false, 0,
                 -5.0, 5.0, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4 * 1024, 512, 1, 1, int, int, true, 1, -15, 15,
                 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4 * 1024, 512, 1, 1, int, int, false, 1, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4 * 1024, 512, 1, 1, float, float, true, 1, -5,
                 5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_ADD, 4 * 1024, 512, 1, 1, float, float, false, 1, -5,
                 5, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MIN, 4 * 1024, 512, 1, 1, int, int, true, 1, -15, 15,
                 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 4 * 1024, 512, 1, 1, int, int, false, 1, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 4 * 1024, 512, 1, 1, float, float, true, 1, -5,
                 5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MIN, 4 * 1024, 512, 1, 1, float, float, false, 1, -5,
                 5, 1e-3);

SCAN_BY_KEY_TEST(AF_BINARY_MAX, 4 * 1024, 512, 1, 1, int, int, true, 1, -15, 15,
                 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 4 * 1024, 512, 1, 1, int, int, false, 1, -15,
                 15, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 4 * 1024, 512, 1, 1, float, float, true, 1, -5,
                 5, 1e-3);
SCAN_BY_KEY_TEST(AF_BINARY_MAX, 4 * 1024, 512, 1, 1, float, float, false, 1, -5,
                 5, 1e-3);

TEST(ScanByKey, Test_Scan_By_key_Simple_0) {
    dim4 dims(16, 8, 2, 1);
    int scanDim = 0;
    int nodel[] = {4, 8};
    vector<int> nodeLengths(nodel, nodel + sizeof(nodel) / sizeof(int));
    int keyStart  = 0;
    int keyEnd    = 15;
    int dataStart = 2;
    int dataEnd   = 4;
    scanByKeyTest<int, int, AF_BINARY_ADD, false>(
        dims, scanDim, nodeLengths, keyStart, keyEnd, dataStart, dataEnd, 1e-5);
}

TEST(ScanByKey, Test_Scan_By_key_Simple_1) {
    dim4 dims(8, 256 + 128, 1, 1);
    int scanDim = 1;
    int nodel[] = {4, 8};
    vector<int> nodeLengths(nodel, nodel + sizeof(nodel) / sizeof(int));
    int keyStart  = 0;
    int keyEnd    = 15;
    int dataStart = 2;
    int dataEnd   = 4;
    scanByKeyTest<int, int, AF_BINARY_ADD, false>(
        dims, scanDim, nodeLengths, keyStart, keyEnd, dataStart, dataEnd, 1e-5);
}

TEST(ScanByKey, FixOverflowWrite) {
    const int SIZE = 41000;
    vector<int> keys(SIZE, 0);
    vector<float> vals(SIZE, 1.0f);

    array someVals = array(SIZE, vals.data());
    array keysAF   = array(SIZE, s32);
    array valsAF   = array(SIZE, vals.data());

    keysAF = array(SIZE, keys.data());

    float prior = valsAF(0).scalar<float>();

    array result;
    try { result = af::scanByKey(keysAF, someVals, 0, AF_BINARY_ADD, true);
    } catch FUNCTION_UNSUPPORTED

    ASSERT_EQ(prior, valsAF(0).scalar<float>());
}
