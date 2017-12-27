Introduction to Vectorization {#vectorization}
===================

Programmers and Data Scientists want to take advantage of fast and parallel
computational devices. Writing vectorized code is necessary to get
the best performance out of the current generation parallel hardware and
scientific computing software. However, writing vectorized code may not be
immediately intuitive. ArrayFire provides many ways to vectorize a given code
segment. In this tutorial, we present several methods to vectorize code
using ArrayFire and discuss the benefits and drawbacks associated with each method.

# Generic/Default vectorization

By its very nature, ArrayFire is a vectorized library. Most functions operate on
arrays as a whole -- on all elements in parallel. Wherever possible, existing
vectorized functions should be used opposed to manually indexing into arrays.
For example consider the following code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10); // [0,  9]
for(int i = 0; i < a.dims(0); ++i)
{
    a(i) = a(i) + 1;         // [1, 10]
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although completely valid, the code is very inefficient as it results in
a kernel kernels that operate on one datum.
Instead, the developer should have used ArrayFire's overload of the + operator:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10);  // [0,  9]
a = a + 1;                    // [1, 10]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code will result in a single kernel that operates on all 10 elements
of `a` in  parallel.

Most ArrayFire functions are vectorized. A small subset of these include:

Operator Category                                           | Functions
------------------------------------------------------------|--------------------------
[Arithmetic operations](\ref arith_mat)                     | [+](\ref arith_func_add), [-](\ref arith_func_sub), [*](\ref arith_func_mul), [/](\ref arith_func_div), [%](\ref arith_func_mod), [>>](\ref arith_func_shiftr), [<<](\ref arith_func_shiftl)
[Logical operations](\ref logic_mat)                        | [&&](\ref arith_func_and), \|\|[(or)](\ref arith_func_or), [<](\ref arith_func_lt), [>](\ref arith_func_gt), [==](\ref arith_func_eq), [!=](\ref arith_func_neq) etc.
[Numeric functions](\ref numeric_mat)                       | abs(), floor(), round(), min(), max(), etc.
[Complex operations](\ref complex_mat)                      | real(), imag(), conj(), etc.
[Exponential and logarithmic functions](\ref explog_mat)    | exp(), log(), expm1(), log1p(), etc.
[Trigonometric functions](\ref trig_mat)                    | sin(), cos(), tan(), etc.
[Hyperbolic functions](\ref hyper_mat)                      | sinh(), cosh(), tanh(), etc.

In addition to element-wise operations, many other functions are also
vectorized in ArrayFire.

Notice that even that perform some form of aggregation (e.g. `sum()` or `min()`),
signal processing (like `convolve()`), and even image processing functions
(i.e. `rotate()`) all support vectorization on different columns or images.
For example, if we have `NUM` images of size `WIDTH` by `HEIGHT`, one could
convolve each image in a vector fashion as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
float g_coef[] = { 1, 2, 1,
                   2, 4, 2,
                   1, 2, 1 };

af::array filter = 1.f/16 * af::array(3, 3, g_coef);

af::array signal = randu(WIDTH, HEIGHT, NUM);
af::array conv = convolve2(signal, filter);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, one can rotate 100 images by 45 degrees in a single call using
code like the following:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// Construct an array of 100 WIDTH x HEIGHT images of random numbers
af::array imgs = randu(WIDTH, HEIGHT, 100);
// Rotate all of the images in a single command
af::array rot_imgs = rotate(imgs, 45);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although *most* functions in ArrayFire do support vectorization, some do not.
Most notably, all linear algebra functions. Even though they are not vectorized
linear algebra operations still execute in parallel on your hardware.

Using the built in vectorized operations should be the first
and preferred method of vectorizing any code written with ArrayFire.

# GFOR: Parallel for-loops

Another novel method of vectorization present in ArrayFire is the GFOR loop
replacement construct. GFOR allows launching all iterations of a loop in parallel
on the GPU or device, as long as the iterations are independent. While the
standard for-loop performs each iteration sequentially, ArrayFire's gfor-loop
performs each iteration at the same time (in parallel). ArrayFire does this by
tiling out the values of all loop iterations and then performing computation on
those tiles in one pass. You can think of gfor as performing auto-vectorization
of your code, e.g. you write a gfor-loop that increments every element of a vector
but behind the scenes ArrayFire rewrites it to operate on the entire vector in
parallel.

The original for-loop example at the beginning of this document could be
rewritten using GFOR as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10);
gfor(seq i, n)
    a(i) = a(i) + 1;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, each instance of the gfor loop is independent, thus ArrayFire
will automatically tile out the `a` array in device memory and execute the
increment kernels in parallel.

To see another example, you could run an accum() on every slice of a matrix in a
for-loop, or you could "vectorize" and simply do it all in one gfor-loop operation:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// runs each accum() in sequence
for (int i = 0; i < N; ++i)
   B(span,i) = accum(A(span,i));

// runs N accums in parallel
gfor (seq i, N)
   B(span,i) = accum(A(span,i));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, returning to our previous vectorization technique, accum() is already
vectorized and the operation could be completely replaced with merely:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    B = accum(A);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is best to vectorize computation as much as possible to avoid the overhead in
both for-loops and gfor-loops. However, the gfor-loop construct is most effective
in the narrow case of broadcast-style operations. Consider the case when we have
a vector of constants that we wish to apply to a collection of variables, such as
expressing the values of a linear combination for multiple vectors. The broadcast
of one set of constants to many vectors works well with gfor-loops:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
const static int p=4, n=1000;
af::array consts = af::randu(p);
af::array var_terms = randn(p, n);

gfor(seq i, n)
    combination(span, i) = consts * var_terms(span, i);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using GFOR requires following several rules and multiple guidelines for optimal
performance. The details of this vectorization method can be found in the
[GFOR documentation](\ref gfor).

# Batching

The batchFunc() function allows the broad application of existing ArrayFire
functions to multiple sets of data. Effectively, batchFunc() allows ArrayFire
functions to execute in "batch processing" mode. In this mode, functions will
find a dimension which contains "batches" of data to be processed and will
parallelize the procedure.

Consider the following example. Here we create a filter which we would like
to apply to each of the weight vectors. The naive solution would be using a
for-loop as we have seen previously:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// Create the filter and the weight vectors
af::array filter = randn(1, 5);
af::array weights = randu(5, 5);

// Apply the filter using a for-loop
af::array filtered_weights = constant(0, 5, 5);
for(int i=0; i<weights.dims(1); ++i){
    filtered_weights.col(i) = filter * weights.col(i);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, as we have discussed above, this solution will be very inefficient.
One may be tempted to implement a vectorized solution as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// Create the filter and the weight vectors
af::array filter = randn(1, 5);
af::array weights = randu(5, 5);

af::array filtered_weights = filter * weights; // fails due to dimension mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, the dimensions of `filter` and `weights` do not match, thus ArrayFire
will generate a runtime error.

`batchfunc()` was created to solve this specific problem.
The signature of the function is as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array batchFunc(const array &lhs, const array &rhs, batchFunc_t func);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

where `__batchFunc_t__` is a function pointer of the form:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
typedef array (*batchFunc_t) (const array &lhs, const array &rhs);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So, to use batchFunc(), we need to provide the function we wish to apply as a
batch operation. For illustration's sake, let's "implement" a multiplication
function following the format.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array my_mult (const af::array &lhs, const af::array &rhs){
    return lhs * rhs;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our final batch call is not much more difficult than the ideal
syntax we imagined.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// Create the filter and the weight vectors
af::array filter = randn(1, 5);
af::array weights = randu(5, 5);

// Apply the batch function
af::array filtered_weights = batchFunc( filter, weights, my_mult );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batch function will work with many previously mentioned vectorized ArrayFire
functions. It can even work with a combination of those functions if they are
wrapped inside a helper function matching the `__batchFunc_t__` signature.
One limitation of `batchfunc()` is that it cannot be used from within a
`gfor()` loop at the present time.

# Advanced Vectorization

We have seen the different methods ArrayFire provides to vectorize our code. Tying
them all together is a slightly more involved process that needs to consider data
dimensionality and layout, memory usage, nesting order, etc. An excellent example
and discussion of these factors can be found on our blog:

http://arrayfire.com/how-to-write-vectorized-code/

It's worth noting that the content discussed in the blog has since been transformed
into a convenient af::nearestNeighbour() function. Before writing something from
scratch, check that ArrayFire doesn't already have an implementation. The default
vectorized nature of ArrayFire and an extensive collection of functions will
speed things up in addition to replacing dozens of lines of code!

