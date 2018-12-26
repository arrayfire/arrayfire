/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>

TEST(Sparse, ReadRealMTXFile) {
    af::array out;
    std::string file(TEST_DIR "/matrixmarket/HB/bcsstm02/bcsstm02.mtx");
    ASSERT_TRUE(mtxReadSparseMatrix(out, file.c_str()));
}

TEST(Sparse, ReadComplexMTXFile) {
    af::array out;
    std::string file(TEST_DIR "/matrixmarket/HB/young4c/young4c.mtx");
    ASSERT_TRUE(mtxReadSparseMatrix(out, file.c_str()));
}

TEST(Sparse, FailIntegerMTXRead) {
    af::array out;
    std::string file(TEST_DIR "/matrixmarket/JGD_Kocay/Trec4/Trec4.mtx");
    ASSERT_FALSE(mtxReadSparseMatrix(out, file.c_str()));
}
