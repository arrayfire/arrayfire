
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
#include <vector>

using namespace af;
using namespace std;

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_tile)
{
    //! [ex_matrix_manipulation_tile]
    float h[] = {1, 2, 3, 4};
    array small_arr = array(2, 2, h); // 2x2 matrix
    af_print(small_arr);
    array large_arr = tile(small_arr, 2, 3);  // produces 4x6 matrix: (2*2)x(2*3)
    af_print(large_arr);
    //! [ex_matrix_manipulation_tile]

    ASSERT_EQ(4, large_arr.dims(0));
    ASSERT_EQ(6, large_arr.dims(1));

    vector<float> h_large_arr(large_arr.elements());
    large_arr.host(&h_large_arr.front());

    unsigned fdim = large_arr.dims(0);
    unsigned sdim = large_arr.dims(1);
    for(unsigned i = 0; i < sdim; i++) {
        for(unsigned j = 0; j < fdim; j++) {
            ASSERT_FLOAT_EQ(h[(i%2) * 2 + (j%2)], h_large_arr[i * fdim + j] );
        }
    }
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_join)
{

    //! [ex_matrix_manipulation_join]
    float hA[] = { 1, 2, 3, 4, 5, 6 };
    float hB[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
    array A = array(3, 2, hA);
    array B = array(3, 3, hB);

    af_print(join(1, A, B)); // 3x5 matrix
    // array result = join(0, A, B); // fail: dimension mismatch
    //! [ex_matrix_manipulation_join]

    array out = join(1, A, B);
    vector<float> h_out(out.elements());
    out.host(&h_out.front());
    af_print(out);

    ASSERT_EQ(3, out.dims(0));
    ASSERT_EQ(5, out.dims(1));

    unsigned fdim = out.dims(0);
    unsigned sdim = out.dims(1);
    for(unsigned i = 0; i < sdim; i++) {
        for(unsigned j = 0; j < fdim; j++) {
            if( i < 2 ) {
                ASSERT_FLOAT_EQ(hA[i * fdim + j], h_out[i * fdim + j]) << "At [" << i << ", " << j << "]";
            }
            else {
                ASSERT_FLOAT_EQ(hB[(i - 2) * fdim + j], h_out[i * fdim + j]) << "At [" << i << ", " << j << "]";
            }
        }
    }

}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_mesh)
{
    //! [ex_matrix_manipulation_mesh]
    float hx[] = {1, 2, 3, 4};
    float hy[] = {5, 6};

    array x = array(4, hx);
    array y = array(2, hy);

    af_print(tile(x, 1, 2));
    af_print(tile(y.T(), 4, 1));
    //! [ex_matrix_manipulation_mesh]

    array outx = tile(x, 1, 2);
    array outy = tile(y.T(), 4, 1);

    ASSERT_EQ(4, outx.dims(0));
    ASSERT_EQ(4, outy.dims(0));
    ASSERT_EQ(2, outx.dims(1));
    ASSERT_EQ(2, outy.dims(1));

    vector<float> houtx(outx.elements());
    outx.host(&houtx.front());
    vector<float> houty(outy.elements());
    outy.host(&houty.front());

    for(unsigned i = 0; i < houtx.size(); i++) ASSERT_EQ(hx[i%4], houtx[i]) << "At [" << i << "]";
    for(unsigned i = 0; i < houty.size(); i++) ASSERT_EQ(hy[i>3], houty[i]) << "At [" << i << "]";
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_moddims)
{
    //! [ex_matrix_manipulation_moddims]
    int hA[] = {1, 2, 3, 4, 5, 6};
    array A = array(3, 2, hA);

    af_print(A); // 2x3 matrix
    af_print(moddims(A, 2, 3)); // 2x3 matrix
    af_print(moddims(A, 6, 1)); // 6x1 column vector

    // moddims(A, 2, 2); // fail: wrong number of elements
    // moddims(A, 8, 8); // fail: wrong number of elements
    //! [ex_matrix_manipulation_moddims]
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_transpose)
{
    //! [ex_matrix_manipulation_transpose]
    array x = randu(2, 2, f32);
    af_print(x.T());  // transpose (real)

    array c = randu(2, 2, c32);
    af_print(c.T());  // transpose (complex)
    af_print(c.H());  // Hermitian (conjugate) transpose
    //! [ex_matrix_manipulation_transpose]
}
