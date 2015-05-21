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

TEST(GettingStarted, SNIPPET_getting_started_gen)
{

    //! [ex_getting_started_gen]
    array zeros      = constant(0, 3);
    array ones       = constant(1, 3, 2, f64);
    array rand1      = randu(1, 4);
    array rand2      = randn(2, 2);
    array iden       = identity(3, 3);
    array randcplx   = randu(2, 1, c32);
    //! [ex_getting_started_gen]

    {
        vector<float> output;
        output.resize(zeros.elements());
        zeros.host(&output.front());
        ASSERT_EQ(f32, zeros.type());
        for(unsigned i = 0; i < zeros.elements(); i++) ASSERT_FLOAT_EQ(0, output[i]);
    }

    {
        vector<double> output(ones.elements());
        ones.host(&output.front());
        ASSERT_EQ(f64, ones.type());
        for(unsigned i = 0; i < ones.elements(); i++) ASSERT_FLOAT_EQ(1, output[i]);
    }

    {
        vector<float> output;
        output.resize(iden.elements());
        iden.host(&output.front());
        for(unsigned i = 0; i < iden.dims(0); i++)
            for(unsigned j = 0; j < iden.dims(1); j++)
                if(i == j)  ASSERT_FLOAT_EQ(1, output[i * iden.dims(0) + j]);
                else        ASSERT_FLOAT_EQ(0, output[i * iden.dims(0) + j]);
    }
}

TEST(GettingStarted, SNIPPET_getting_started_init)
{
    //! [ex_getting_started_init]
    float hA[] = {0, 1, 2, 3, 4, 5};
    array A(2, 3, hA);      // 2x3 matrix of single-precision
    af_print(A);            // Note: Fortran storage order (column major)
    //! [ex_getting_started_init]

    vector<float> out(A.elements());
    A.host(&out.front());
    for(unsigned int i = 0; i < out.size(); i++) ASSERT_FLOAT_EQ(hA[i], out[i]);
}

TEST(GettingStarted, SNIPPET_getting_started_print)
{
    //! [ex_getting_started_print]
    array a = randu(2, 2);
    array b = constant(1, 2, 1);
    af_print(a);
    af_print(b);
    af_print(a.col(0) + b + .4);
    //! [ex_getting_started_print]
    array result = a.col(0) + b + 0.4;
    vector<float> outa(a.elements());
    vector<float> outb(b.elements());
    vector<float> out(result.elements());

    a.host(&outa.front());
    b.host(&outb.front());
    result.host(&out.front());

    for(unsigned i = 0; i < outb.size(); i++) ASSERT_FLOAT_EQ(outa[i] + outb[i] + 0.4, out[i]);
}

TEST(GettingStarted, SNIPPET_getting_started_dims)
{
    //! [ex_getting_started_dims]
    array a = randu(4,5,2);
    printf("numdims(a)  %d\n",  a.numdims()); // 3

    dim4 dims = a.dims();
    printf("dims = [%lld %lld]\n", dims[0], dims[1]); // 4,5
    printf("dims = [%lld %lld]\n", a.dims(0), a.dims(1)); // 4,5
    //! [ex_getting_started_dims]

    //! [ex_getting_started_prop]
    printf("underlying type: %d\n", a.type());
    printf("is complex? %d    is real? %d\n", a.iscomplex(), a.isreal());
    printf("is vector? %d  column? %d  row? %d\n", a.isvector(), a.iscolumn(), a.isrow());
    printf("empty? %d  total elements: %lld  bytes: %lu\n", a.isempty(), a.elements(), a.bytes());
    //! [ex_getting_started_prop]

    ASSERT_EQ(f32, a.type());
    ASSERT_TRUE(a.isreal());
    ASSERT_FALSE(a.iscomplex());
    ASSERT_FALSE(a.isvector());
    ASSERT_FALSE(a.iscolumn());
    ASSERT_FALSE(a.isrow());
    ASSERT_FALSE(a.isempty());
    ASSERT_EQ(40, a.elements());

    ASSERT_EQ(f32, a.type());
    ASSERT_EQ(f32, a.type());

    ASSERT_EQ(4, dims[0]);
    ASSERT_EQ(4, a.dims(0));
    ASSERT_EQ(5, dims[1]);
    ASSERT_EQ(5, a.dims(1));
}

TEST(GettingStarted, SNIPPET_getting_started_arith)
{
    //! [ex_getting_started_arith]
    array R = randu(3, 3);
    af_print(constant(1, 3, 3) + af::complex(sin(R)));  // will be c32

    // rescale complex values to unit circle
    array a = randn(5, c32);
    af_print(a / abs(a));

    // calculate L2 norm of vectors
    array X = randn(3, 4);
    af_print(sqrt(sum(pow(X, 2))));     // norm of every column vector
    af_print(sqrt(sum(pow(X, 2), 0)));  // same as above
    af_print(sqrt(sum(pow(X, 2), 1)));  // norm of every row vector
    //! [ex_getting_started_arith]
}

TEST(GettingStarted, SNIPPET_getting_started_dev_ptr)
{
#ifdef __CUDACC__
    //! [ex_getting_started_dev_ptr]
    float host_ptr[] = {0,1,2,3,4,5}; // column-major order (like Fortran)
    array a(2, 3, host_ptr); // f32 matrix of size 2-by-3 from host data

    float *device_ptr;
    cudaMalloc((void**)&device_ptr, 6*sizeof(float));
    cudaMemcpy(device_ptr, host_ptr, 6*sizeof(float), cudaMemcpyHostToDevice);
    array b(2,3, host_ptr, afDevice); // Note: afDevice (default: afHost)
    // do not call cudaFree(device_ptr) -- it is freed when 'b' is destructed.

    // create complex data
    cuComplex ha[] = { {0,1}, {2,3}, {4,5} }; // { {real,imaginary}, {real,imag}, .. }
    array a(3,ha); // 3x1 column vector
    //! [ex_getting_started_dev_ptr]
#endif //__CUDACC__
}

TEST(GettingStarted, SNIPPET_getting_started_ptr)
{
#ifdef __CUDACC__
    //! [ex_getting_started_ptr]
    array a = randu(3, f32);
    float *host_a = a.host<float>();        // must call array::free() later
    printf("host_a[2] = %g\n", host_a[2]);  // last element
    array::free(host_a);

    float *device_a = a.device<float>();    // no need to free this
    float value;
    cudaMemcpy(&value, device_a + 2, sizeof(float), cudaMemcpyDeviceToHost);
    printf("device_a[2] = %g\n", value);
    a.unlock(); // unlock to allow garbage collection if necessary
    //! [ex_getting_started_ptr]
#endif //__CUDACC__
}


TEST(GettingStarted, SNIPPET_getting_started_scalar)
{
    //! [ex_getting_started_scalar]
    array a = randu(3);
    float val = a.scalar<float>();
    printf("scalar value: %g\n", val);
    //! [ex_getting_started_scalar]
}

TEST(GettingStarted, SNIPPET_getting_started_bit)
{
    //! [ex_getting_started_bit]
    int h_A[] = {1, 1, 0, 0, 4, 0, 0, 2, 0};
    int h_B[] = {1, 0, 1, 0, 1, 0, 1, 1, 1};
    array A = array(3, 3, h_A), B = array(3, 3, h_B);
    af_print(A); af_print(B);

    array A_and_B = A & B; af_print(A_and_B);
    array  A_or_B = A | B; af_print(A_or_B);
    array A_xor_B = A ^ B; af_print(A_xor_B);
    //! [ex_getting_started_bit]

    vector<int> Andout(A_and_B.elements());
    vector<int> Orout(A_or_B.elements());
    vector<int> Xorout(A_xor_B.elements());
    A_and_B.host(&Andout.front());
    A_or_B.host(&Orout.front());
    A_xor_B.host(&Xorout.front());


    for(unsigned int i = 0; i < Andout.size(); i++) ASSERT_FLOAT_EQ(h_A[i] & h_B[i], Andout[i]);
    for(unsigned int i = 0; i < Orout.size(); i++)  ASSERT_FLOAT_EQ(h_A[i] | h_B[i], Orout[i]);
    for(unsigned int i = 0; i < Xorout.size(); i++) ASSERT_FLOAT_EQ(h_A[i] ^ h_B[i], Xorout[i]);
}


TEST(GettingStarted, SNIPPET_getting_started_constants)
{
    //! [ex_getting_started_constants]
    array A = randu(5,5);
    A(where(A > .5)) = af::NaN;

    array x = randu(10e6), y = randu(10e6);
    double pi_est = 4 * sum<float>(hypot(x,y) < 1) / 10e6;
    printf("estimation error: %g\n", fabs(Pi - pi_est));
    //! [ex_getting_started_constants]

    ASSERT_LE(fabs(Pi-pi_est), 0.0005);
}

