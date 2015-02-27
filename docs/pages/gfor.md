GFOR: Parallel For-Loops {#page_gfor}
========================

[TOC]

Run many independent loops simultaneously on the GPU or device.

Introduction {#gfor_intro}
============

The gfor-loop construct may be used to simultaneously launch all of
the iterations of a for-loop on the GPU or device, as long as the
iterations are independent. While the standard for-loop performs each
iteration sequentially, ArrayFire's gfor-loop performs each iteration
at the same time (in parallel). ArrayFire does this by tiling out the
values of all loop iterations and then performing computation on those
tiles in one pass.

You can think of `gfor` as performing auto-vectorization of your
code, e.g. you write a gfor-loop that increments every element of a
vector but behind the scenes ArrayFire rewrites it to operate on
the entire vector in parallel.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
for (int i = 0; i < n; ++i)
   A(i) = A(i) + 1;

gfor (array i, n)
   A(i) = A(i) + 1;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Behind the scenes, ArrayFire rewrites your code into this
equivalent and faster version:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = A + 1;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is best to vectorize computation as much as possible to avoid
the overhead in both for-loops and gfor-loops.

To see another example, you could run an FFT on every 2D slice of a
volume in a for-loop, or you could "vectorize" and simply do it all
in one gfor-loop operation:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
for (int i = 0; i < N; ++i)
   A(span,span,i) = fft2(A(span,span,i)); // runs each FFT in sequence

gfor (array i, N)
   A(span,span,i) = fft2(A(span,span,i)); // runs N FFTs in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are three formats for instantiating gfor-loops.
-# gfor(var,n) Creates a sequence _{0, 1, ..., n-1}_
-# gfor(var,first,last) Creates a sequence _{first, first+1, ..., last}_
-# gfor(var,first,incr,last) Creates a sequence _{first, first+inc, first+2*inc, ..., last}_

So all of the following represent the equivalent sequence: _0,1,2,3,4_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array i, 5)
gfor (array i, 0, 4)
gfor (array i, 0, 1, 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More examples:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = constant(1,n,n);
array B = constant(1,1,n);
gfor (array k, 0, n-1) {
   B(k) = A(k,span) * A(span,k);  // inner product
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = constant(1,n,n,m);
array B = constant(1,n,n);
gfor (array k, 0,m-1) {
   A(span,span,k) = A(span,span,k) * B; // matrix-matrix multiply
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = randu(n,m);
array B = constant(0,n,m);
gfor (array k, 0, m-1) {
   B(span,k) = fft(A(span,k));
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage {#gfor}
=====

User Functions called within GFOR {#gfor_user_functions}
---------------------------------

If you have defined a function that you want to call within a GFOR loop, then
that function has to meet all the conditions described in this page in
order to be able to work as expected.

Consider the (trivial) example below. The function compute() has to satisfy all
requirements for GFOR Usage, so you cannot use if-else conditions inside
it.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array compute(array A, array B, float ep)
{
  array H;
  if (ep > 0) H = (A * B) / ep;  // BAD
  else        H = A * 0;
  return H;
}

int m = 2, n = 3;
array A = randu(m, n);
array B = randu(m, n);
float ep = 2.35;
array H = constant(0,m,n);
gfor (array ii, n)
  H(span,ii) = compute(A(span,ii), B(span,ii), ep);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiplications {#gfor_mul}
---------------

ArrayFire supports bulk multiplications of vector-vector, matrix-vector, and
matrix-matrix types using GFOR. This is especially useful with many small
matrices.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = constant(1,n,n);
array B = constant(1,n,1);
array C = constant(0,n,m);
gfor (array k, n)
  B(k) = A(k,span) * A(span,k); // vector-vector multiply

A = constant(1,n,n,m);
gfor (array k, m)
  C(span,k) = A(span,span,k) * B;  // matrix-vector multiply

A = constant(1,n,n,m);
B = constant(1,n,n);
gfor (array k, m)
  A(span,span,k) = A(span,span,k) * B;  // matrix-matrix multiply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Iterator {#gfor_iterator}
------------

The iterator can be involved in expressions.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n,m);
B = constant(1,n,n);
gfor (array k, m)
  A(span,span,k) = (k+1)*B + sin(k+1);  // expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterator definitions can include arithmetic in expressions.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n,m);
B = constant(1,n,n);
gfor (array k, m/4, m-m/4)
  A(span,span,k) = k*B + sin(k+1);  // expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subscripting {#gfor_subscripting}
------------

More complicated subscripting is supported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n,m);
B = constant(1,n,10);
gfor (array k, m)
  A(span,seq(10),k) = k*B;  // subscripting, seq(10) generates index [0,9]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterators can be combined with arithmetic in subscripts.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = randu(n,m);
array B = constant(1,n,m);
gfor (array k, 1, m-1)
  B(span,k) = A(span,k-1);

A = randu(n,2*m);
B = constant(1,n,m);
gfor (array k, m)
  B(span,k) = A(span,2*(k+1)-1);

A = randu(n,2*m);
B = constant(1,n,m);
gfor (array k, m)
  B(span,k) = A(span,floor(k+.2));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In-Loop Reuse {#gfor_in_loop}
-------------

Within the loop, you can use a result you just computed.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array k, n) {
  A(span,k) = 4 * B(span,k);
  C(span,k) = 4 * A(span,k); // use it again
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although it is more efficient to store the value in a temporary variable:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array k, n) {
  a = 4 * B(span,k);
  A(span,k) = a;
  C(span,k) = 4 * a;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\note If the variable a above had not involved a GFOR expression, you may need to use local() to designate it as a temporary variable specific to each iteration.

In-Place Computation {#gfor_in_place_computation}
--------------------

In some cases, GFOR behaves differently than the typical sequential
FOR-loop. For example, you can read and modify a result in place as long as
the accesses are independent.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n);
gfor (array k, n)
  A(span,k) = sin(k) + A(span,k);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subscripting behaviors `arrays` also work with GFOR.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n,m,k);
m = m * k; // precompute since cannot have expressions in iterator
gfor (array k, m)
  A(span,span,k) = 4 * A(span,span,k); // collapse last dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Random Data Generation {#gfor_random}
----------------------

Random data should always be generated outside the GFOR loop. This is due to
the fact that GFOR only passes over the body of the loop once. Therefore,
any calls to randu() inside the body of the loop will result in the same
random matrix being assigned to every iteration of the loop.

For example, in the following trivial code, all columns of `B` are identical
because `A` is only evaluated once:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array ii, n) {
  array A = randu(3,1);
  B(span,ii) = A;
}
print(B);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	B =
	   0.1209    0.1209    0.1209
	   0.6432    0.6432    0.6432
	   0.8746    0.8746    0.8746

This can be rectified by bringing the random number generation outside the
loop, as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = randu(3,n);
gfor (array ii, n)
  B(span,ii) = A(span,ii);
print(B);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	B =
	  0.0892    0.1655    0.7807
	  0.5626    0.5173    0.2932
	  0.5664    0.5898    0.1391

This is a trivial example, but demonstrates the principle that random numbers
should be pre-allocated outside the loop in most cases.

Local variables {#gfor_local_variables}
---------------

Use local() to create local copies of data that can be modified
independently in each GFOR tile. In general, in a subscript assignment
involving the iterator among the subscripts, ArrayFire assumes you are writing a
final result out from the GFOR-loop. However, in many cases you may want to
subscript assign results to a variable where each iteration (tile) has its own
uniquely-modified copy of that variable.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = constant(1,n,m);
array B = constant(1,A.dims());
gfor (array k, m) { // 0, 1, ..., m-1
   array a = A(span,k); // local() not needed since iterator subscripting
   array b = local(constant(0,n,1)); // local() needed
   b(seq(2)) = a(seq(2));  // each GFOR tile gets its own unique copy of 'b'
   B(span,k) = b;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrictions {#gfor_restrictions}
============

This preliminary implementation of GFOR has the following restrictions.

Iteration independence {#gfor_iteration_independence}
----------------------

The most important property of the loop body is that each iteration must be
independent of the other iterations. Note that accessing the result of a
separate iteration produces undefined behavior.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array B = randu(3);
gfor (array k, n)
  B = B + k; // bad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No conditional statements {#gfor_no_cond}
-------------------------

No conditional statements in the body of the loop, (i.e. no
branching). However, you can often find ways to overcome this
restriction. Consider the following two examples:

Example 1: Problem

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,m);
gfor (array k, n) {
 if (k > 10) A(span,k) = k + 1; // bad
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, you can do a few tricks to overcome this restriction by expressing
the conditional statement as a multiplication by logical values. For instance,
the block of code above can be converted to run as follows, without error:

Example 1: Solution

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array k, m) {
  array condition = (k > 1); // good
  A(span,k) = (!condition).as(f32) * A(span,k) + condition.as(f32) * (k + 1);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another example of overcoming the conditional statement restriction in GFOR is
as follows:

Example 2: Problem

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
array A = constant(1,n,n,m);
array B = randu(n,n);
gfor (array k, 4) {
  if ((k % 2) != 0)
	 A(span,span,k) = B + k;
  else
	 A(span,span,k) = B * k;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead, you can make two passes over the same data, each pass performing one
branch.

Example 2: Solution

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
A = constant(1,n,n,m);
B = randu(n);
gfor (array k, 0, 2, 3)
  A(span,span,k) = B + k;
gfor (array k, 1, 2, 3)
  A(span,span,k) = B * k;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nested loop restrictions {#gfor_nested_loop}
------------------------

Nesting GFOR-loops within GFOR-loops is unsupported. You may interleave
FOR-loops as long as they are completely independent of the GFOR-loop
iterator.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array k, n) {
  gfor (array j, m) { // bad
  // ...
  }
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nesting FOR-loops within GFOR-loops is supported, as long as the GFOR iterator
is not used in the FOR loop iterator, as follows:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array k, n) {
  for (int j = 0; j < (m+k); j++) { // bad
  // ...
  }
}

gfor (array k, n) {
  for (int j = 0; j < m; j++) { // good
  //...
  }
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nesting GFOR-loops inside of FOR-loops is fully supported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
for (array k, n) {
  gfor (int j = 0; j < m; j++) { // good
  //  ...
  }
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No logical indexing {#gfor_no_logical}
-------------------

Logical indexing like the following is not supported:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array i, n) {
  array B = A(span,i);
  array tmp = B(B > .5); // bad
  D(i) = sum(tmp);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The problem is that every GFOR tile has a different number of elements, something which GFOR cannot yet handle.

Similar to the workaround for conditional statements, it might work to use
masked arithmetic:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array i, n) {
  array B = A(span,i);
  array mask = B > .5;
  D(i) = sum(mask .* B);
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sub-assignment with scalars and logical masks is supported:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
gfor (array i, n) {
  a = A(span,i);
  a(isnan(a)) = 0;
  A(span,i) = a;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory considerations {#gfor_memory}
=====================

Since each computation is done in parallel for all iterator values,
you need to have enough card memory available to do all iterations
simultaneously. If the problem exceeds memory, it will trigger "out of
memory" errors.

You can work around the memory limitations of your GPU or device by
breaking the GFOR loop up into segments; however, you might want to
consider using a larger memory GPU or device.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// BEFORE
gfor (array k, 400) {
  array B = A(span,k);
  C(span,span,k) = B * B.T();  // outer product expansion runs out of memory
}

// AFTER
for (int kk = 0; kk < 400; kk += 100) {
  gfor (array k, kk, kk+99) { // four batches of 100
	 array B = A(span,k);
	 C(span,span,k) = B * B.T(); // now several smaller problems fit in card memory
  }
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


