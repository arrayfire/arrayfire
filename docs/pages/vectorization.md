Vectorization {#vectorization}
===================

Programmers and Data Scientists want to take advantage of fast and parallel computational devices. Writing vectorized code is becoming a necessity to get the best performance out of the current generation parallel hardware and scientific computing software. However, writing vectorized code may not be intuitive immediately. Arrayfire provides many ways to vectorize a given code segment. In this tutorial, we will be presenting various ways to vectorize code using ArrayFire and the benefits and drawbacks associated with each method.

# Generic/Default vectorization
By its very nature, Arrayfire is a vectorized library. Most functions operate on arrays as a whole -- on all elements in parallel. Wherever possible, existing vectorized functions should be used opposed to manually indexing into arrays. For example, consider this valid, yet mislead code that attempts to increment each element of an array:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::seq(10); // [0,  9]
for(int i=0; i<a.dims(0); ++i)
{
    a(i) = a(i) + 1;       // [1, 10]
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, the existing vectorized Arrayfire overload of the + operator should have been used:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::seq(10);  // [0,  9]
a = a + 1;                  // [1, 10]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the vectorized mathematical functions of Arrayfire include:

Operator Category                     | Functions
--------------------------------------|--------------------------
Arithmetic operations                 | operator+(), operator-(), operator*(), operator/(), operator>>(), operator<<()
Complex operations                    | real(), imag(), conjugate(), etc.
Exponential and logarithmic functions | exp(), log(), expm1(), log1p(), etc.
Hyperbolic functions                  | sinh(), cosh(), tanh(), etc.
Numeric functions                     | floor(), round(), min(), max(), etc.
Trigonometric functions               | sin(), cos(), tan(), etc.
Logical operations                    | &&, \|\|, \|, &, <, >, <=, >=, ==, !


# GFOR: Parallel for-loops
Another novel method of vectorization present in Arrayfire is the GFOR loop replacement construct.
GFOR allows launching all iterations of a loop in parallel on the GPU or device, as long as the iterations are independent. While the standard for-loop performs each iteration sequentially, ArrayFire's gfor-loop performs each iteration at the same time (in parallel). ArrayFire does this by tiling out the values of all loop iterations and then performing computation on those tiles in one pass.
You can think of gfor as performing auto-vectorization of your code, e.g. you write a gfor-loop that increments every element of a vector but behind the scenes ArrayFire rewrites it to operate on the entire vector in parallel.
We can remedy our first example with GFOR:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::seq(10);
gfor(seq i, n)
    a(i) = a(i) + 1;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is best to vectorize computation as much as possible to avoid the overhead in both for-loops and gfor-loops.

To see another example, you could run an FFT on every 2D slice of a volume in a for-loop, or you could "vectorize" and simply do it all in one gfor-loop operation:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
for (int i = 0; i < N; ++i)
   A(span,span,i) = fft2(A(span,span,i)); // runs each FFT in sequence
gfor (seq i, N)
   A(span,span,i) = fft2(A(span,span,i)); // runs N FFTs in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## GFOR: Usage
There are three formats for instantiating gfor-loops:

 1. gfor(var,n)-- Creates a sequence <B>{0, 1, ..., n-1}</B>
 2. gfor(var,first,last)-- Creates a sequence <B>{first, first+1, ..., last}</B>
 3. gfor(var,first,incr,last)-- Creates a sequence <B>{first, first+inc, first+2 * inc, ..., last}</B>


All of the following represent the equivalent sequence: 0,1,2,3,4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (seq i, 5)
gfor (seq i, 0, 4)
gfor (seq i, 0, 1, 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using GFOR requires following several rules and multiple guidelines for optimal performance. The details of this vectorization method can be found in the <a href="page_gfor.htm">GFOR documentation.</a>

# batchFunc()
The batchFunc() function allows the broad application of existing Arrayfire functions to multiple sets of data. Effectively, batchFunc() allows Arrayfire functions to execute in "batch processing" mode. In this mode, functions will find a dimension which contains "batches" of data to be processed and will parallelize the procedure.
Consider the following example:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array filter = randn(1, 5);
af::array weights = randu(5, 5);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have a filter that we would like to apply to each of several weights vectors.
The naive solution would be using a loop as we've seen before:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array filtered_weights = constant(0, 5, 5);
for(int i=0; i<weights.dims(1); ++i){
    filtered_weights.col(i) = filter * weights(i);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
However we would like a vectorized solution. The following syntax begs to be used:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array filtered_weights = filter * weights; //fails due to dimension mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
but it fails due to the (5x1), (5x5) dimension mismatch. Wouldn't it be nice if Arrayfire could figure out along which dimension we intend to apply the batch operation? That is exactly what batchFunc() does!
The signature of the function is:

__AFAPI array batchFunc( const array &lhs, const array &rhs, batchFunc_t func );__

where __batchFunc_t__ is a function pointer of the form:
__typedef array (*batchFunc_t) ( const array &lhs, const array &rhs );__


So, to use batchFunc(), we need to provide the function we will be applying as a batch operation. Our final batch call is not much more difficult than the ideal syntax we imagined.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array filtered_weights = batchFunc(filter, weights, operator* );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batch function will work with many previously mentioned vectorized Arrayfire functions. It can even work with a combination of those functions if they are wrapped inside a helper function matching the __batchFunc_t__ signature. Unfortunately, the batch function cannot be used within a gfor() construct at this moment.

