Indexing {#indexing}
========

Indexing in ArrayFire is a powerful but easy to abuse feature of the af::array
class. This feature allows you to reference or copy subsections of a larger array
and perform operations on only a subset of elements.

Indexing in ArrayFire can be performed using the parenthesis operator or one of
the member functions of the af::array class. These functions allow you to
reference one or a range of elements from the original array.

Here we will demonstrate some of the ways you can use indexing in ArrayFire and
discuss ways to minimize the memory and performance impact of these operations.

Lets start by creating a new 4x4 matrix of floating point numbers:

\snippet test/index.cpp index_tutorial_1

The ArrayFire library is column-major so the resulting A array will
look like this:

\f[
\begin{bmatrix}
    0 & 4 & 8 & 12 \\
    1 & 5 & 9 & 13 \\
    2 & 6 & 10 & 14 \\
    3 & 7 & 11 & 15
\end{bmatrix}
\f]

This is a two dimensional array so we can access the first element of this
matrix by passing `0,0` into the parenthesis operator of the af::array.

\snippet test/index.cpp index_tutorial_first_element

\f[ A(0, 0) = [ 0 ] \f]

We can also access the array using linear indexing by passing in one value. Here
we are accessing the fifth element of the array.

\snippet test/index.cpp index_tutorial_fifth_element

\f[ A(5) = [ 5 ] \f]

Normally you want to avoid accessing individual elements of the array like this
for performance reasons.

## af::span and af::seq

You can access regions regions of the array via the af::seq and af::span
objects. The span objects allows you to select the entire set of elements across
a particular dimension/axis of an array. For example, we can select the third
column of the array by passing span as the first agument and 2 as the second
argument to the parenthesis operator.

\snippet test/index.cpp index_tutorial_third_column

\f[
A(span, 2) =
\begin{bmatrix}
    8 \\
    9 \\
    10 \\
    11
\end{bmatrix}
\f]

You can read that as saying that you want all values across the first dimension,
but only from index 2 of the second dimension.

You can access the second rows by passing (1, span) to the array

\snippet test/index.cpp index_tutorial_second_row

\f[ A(1, span) = [ 1, 5, 9, 13 ] \f]

You can use the af::seq (short for sequence) object defines a range of values
when indexing. For example if you wanted to get the first two colums, you would
access the array by passing span for the first argument and seq(2) as the second
argument.

\snippet test/index.cpp index_tutorial_first_two_columns

\f[
A(span, seq(2)) =
\begin{bmatrix}
     0 & 4 \\
     1 & 5 \\
     2 & 6 \\
     3 & 7
\end{bmatrix}
\f]

The af::seq object can also be used to define ranges that are not continuous. There are
three constructors for af::seq.

* af::seq(N): Defines a range between 0 and N-1
* af::seq(begin, end) Defines a range between begin and end inclusive
* af::seq(begin, end, step) defines a range between begin and end striding by step values

You can select the second and forth rows by passing (seq(1, 4, 2), span) to the
indexing operator.

\snippet test/index.cpp index_tutorial_second_and_fourth_rows

\f[
A(seq(1, 3, 2), span) =
\begin{bmatrix}
     1 & 5 &  9 & 13 \\
     3 & 7 & 11 & 15
\end{bmatrix}
\f]

## Indexing using af::array

You can also index using other af::array objects. ArrayFire performs a cartesian
product of the input arrays.

\snippet test/index.cpp index_tutorial_array_indexing

\f[
A =
\begin{bmatrix}
    0 & 4 & 8 & 12 \\
    1 & 5 & 9 & 13 \\
    2 & 6 & 10 & 14 \\
    3 & 7 & 11 & 15
\end{bmatrix}
\\
A(
\begin{bmatrix}
2 \\ 1 \\ 3
\end{bmatrix}
,
\begin{bmatrix}
3 \\ 1 \\ 2
\end{bmatrix}
) =

\begin{bmatrix}
(2,3) & (2,1) & (2,2) \\
(1,3) & (1,1) & (1,2) \\
(3,3) & (3,1) & (3,2)
\end{bmatrix}
=
\begin{bmatrix}
14 & 6 & 10 \\
13 & 5 &  9 \\
15 & 7 & 11
\end{bmatrix}
\f]


If you want to index an af::array using coordinate arrays, you can do that using the
af::approx1 and af::approx2 functions.

\snippet test/index.cpp index_tutorial_approx

\f[
approx2(A,
\begin{bmatrix}
2 \\ 1 \\ 3
\end{bmatrix}
,
\begin{bmatrix}
3 \\ 1 \\ 2
\end{bmatrix}
) =
\begin{bmatrix}
(2,3) \\
(1,1) \\
(3,2)
\end{bmatrix}
=
\begin{bmatrix}
14 \\
 5 \\
11
\end{bmatrix}
\f]

## References and copies

All ArrayFire indexing functions return af::array(technically its an array_proxy
class but we will discuss that later** objects. These objects may be new arrays
or they may reference the original array depending on the type of indexing that
was performed on them. If an array was indexed using another af::array or it
was indexed using the af::approx functions, then a new array is created. It does
not reference the original data.

If an array was indexed using a scalar, af::seq or af::span, then the resulting
array will reference the original data IF the first dimension is continuous. The
following indexing approaches will not allocate additional memory.

\snippet test/index.cpp index_tutorial_references

The following code snippet shows some examples of indexing that will allocate
new memory.

\snippet test/index.cpp index_tutorial_copies

Notice that even though the copy3 array is refrencing continuous memory in the
original array, a new array is created because we used an array to index into
the af::array.

## Assignment

Even though the new af::array array objects are referencing the original elements,
any modifications you do to them will allocate additional memory.


TODO(umar): Subarray assignment does not change original
TODO(umar): parent assignment does not change subarrays


## Member functions

There are several member functions which allow you to index into an af::array. These
functions have similar functionallity but may be easier to parse for some.

* [row(i)](\ref af::array::row) or [col(i)](\ref af::array::col) specifying a single row/column
* [rows(first,last)](\ref af::array::rows) or [cols(first,last)](\ref af::array::cols)
 specifying a span of rows or columns


# Performance and implementation

TODO(umar): talk about performance

See \ref  index_mat for the full listing.

\snippet test/index.cpp ex_indexing_first

You can set values in an array:

\snippet test/index.cpp ex_indexing_set

Use one array to reference into another.

\snippet test/index.cpp ex_indexing_ref
