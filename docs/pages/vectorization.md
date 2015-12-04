Introduction to Vectorization {#vectorization}
===================

Programmers and Data Scientists want to take advantage of fast and parallel
computational devices. Writing vectorized code is becoming a necessity to get
the best performance out of the current generation parallel hardware and
scientific computing software. However, writing vectorized code may not be
intuitive immediately. Arrayfire provides many ways to vectorize a given code
segment. In this tutorial, we will be presenting various ways to vectorize code
using ArrayFire and the benefits and drawbacks associated with each method.

# Generic/Default vectorization
By its very nature, Arrayfire is a vectorized library. Most functions operate on
arrays as a whole -- on all elements in parallel. Wherever possible, existing
vectorized functions should be used opposed to manually indexing into arrays.
For example, consider this valid, yet mislead code that attempts to increment
each element of an array:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10); // [0,  9]
for(int i = 0; i < a.dims(0); ++i)
{
    a(i) = a(i) + 1;         // [1, 10]
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, the existing vectorized Arrayfire overload of the + operator should have been used:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10);  // [0,  9]
a = a + 1;                    // [1, 10]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most Arrayfire functions are vectorized. A small subset of these include:

Operator Category                                           | Functions
------------------------------------------------------------|--------------------------
[Arithmetic operations](\ref arith_mat)                     | [+](\ref arith_func_add), [-](\ref arith_func_sub), [*](\ref arith_func_mul), [/](\ref arith_func_div), [%](\ref arith_func_mod), [>>](\ref arith_func_shiftr), [<<](\ref arith_func_shiftl)
[Complex operations](\ref complex_mat)                      | real(), imag(), conj(), etc.
[Exponential and logarithmic functions](\ref explog_mat)    | exp(), log(), expm1(), log1p(), etc.
[Hyperbolic functions](\ref hyper_mat)                      | sinh(), cosh(), tanh(), etc.
[Logical operations](\ref logic_mat)                        | [&&](\ref arith_func_and), \|\|[(or)](\ref arith_func_or), [<](\ref arith_func_lt), [>](\ref arith_func_gt), [==](\ref arith_func_eq), [!=](\ref arith_func_neq) etc.
[Numeric functions](\ref numeric_mat)                       | abs(), floor(), round(), min(), max(), etc.
[Trigonometric functions](\ref trig_mat)                    | sin(), cos(), tan(), etc.

In addition to element-wise operations, many other functions are also
vectorized in Arrayfire.

Vector operations such as min() support vectorization:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array arr = randn(100);
std::cout << min<float>(arr) << std::endl;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signal processing functions like convolve() support vectorization:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
float g_coef[] = { 1, 2, 1,
                   2, 4, 2,
                   1, 2, 1 };

af::array filter = 1.f/16 * af::array(3, 3, f_coef);

af::array signal = randu(WIDTH, HEIGHT, NUM);
af::array conv = convolve2(signal, filter);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image processing functions such as rotate() support vectorization:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array imgs = randu(WIDTH, HEIGHT, 100); // 100 (WIDTH x HEIGHT) images
af::array rot_imgs = rotate(imgs, 45); // 100 rotated images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One class of functions that does not support vectorization is the set of linear
algebra functions. Using the built in vectorized operations should be the first
and preferred method of vectorizing any code written with Arrayfire.

# GFOR: Parallel for-loops
Another novel method of vectorization present in Arrayfire is the GFOR loop
replacement construct. GFOR allows launching all iterations of a loop in parallel
on the GPU or device, as long as the iterations are independent. While the
standard for-loop performs each iteration sequentially, ArrayFire's gfor-loop
performs each iteration at the same time (in parallel). ArrayFire does this by
tiling out the values of all loop iterations and then performing computation on
those tiles in one pass. You can think of gfor as performing auto-vectorization
of your code, e.g. you write a gfor-loop that increments every element of a vector
but behind the scenes ArrayFire rewrites it to operate on the entire vector in
parallel.

We can remedy our first example with GFOR:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array a = af::range(10);
gfor(seq i, n)
    a(i) = a(i) + 1;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see another example, you could run an accum() on every slice of a matrix in a
for-loop, or you could "vectorize" and simply do it all in one gfor-loop operation:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
for (int i = 0; i < N; ++i)
   B(span,i) = accum(A(span,i)); // runs each accum() in sequence
gfor (seq i, N)
   B(span,i) = accum(A(span,i)); // runs N accums in parallel
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

Using GFOR requires following several rules and multiple guidelines for optimal performance.
The details of this vectorization method can be found in the [GFOR documentation](\ref gfor).

# Batching
The batchFunc() function allows the broad application of existing Arrayfire
functions to multiple sets of data. Effectively, batchFunc() allows Arrayfire
functions to execute in "batch processing" mode. In this mode, functions will
find a dimension which contains "batches" of data to be processed and will
parallelize the procedure.

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
    filtered_weights.col(i) = filter * weights.col(i);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However we would like a vectorized solution. The following syntax begs to be used:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
af::array filtered_weights = filter * weights; // fails due to dimension mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This fails due to the (5x1), (5x5) dimension mismatch. Wouldn't it be nice if
Arrayfire could figure out along which dimension we intend to apply the batch
operation? That is exactly what batchFunc() does!
The signature of the function is:

`array batchFunc(const array &lhs, const array &rhs, batchFunc_t func);`

where __batchFunc_t__ is a function pointer of the form:
`typedef array (*batchFunc_t) (const array &lhs, const array &rhs);`


So, to use batchFunc(), we need to provide the function we will be applying as a
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
af::array filtered_weights = batchFunc( filter, weights, my_mult );
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batch function will work with many previously mentioned vectorized Arrayfire
functions. It can even work with a combination of those functions if they are
wrapped inside a helper function matching the __batchFunc_t__ signature. Unfortunately,
the batch function cannot be used within a gfor() construct at this moment.

# Advanced Vectorization
We have seen the different methods Arrayfire provides to vectorize our code. Tying
them all together is a slightly more involved process that needs to consider data
dimensionality and layout, memory usage, nesting order, etc. An excellent example
and discussion of these factors can be found on our blog:

http://arrayfire.com/how-to-write-vectorized-code/

It's worth noting that the content discussed in the blog has since been transformed
into a convenient af::nearestNeighbour() function. Before writing something from
scratch, check that Arrayfire doesn't already have an implementation. The default
vectorized nature of Arrayfire and an extensive collection of functions will
speed things up in addition to replacing dozens of lines of code!

