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

ArrayFire is column-major so the resulting A array will look like this:

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

\f[ A(2, 3) = [ 14 ] \f]

We can also access the array using linear indexing by passing in one value. Here
we are accessing the fifth element of the array.

\snippet test/index.cpp index_tutorial_fifth_element

\f[ A(5) = [ 5 ] \f]

\note Normally you want to avoid accessing individual elements of the array like this
for performance reasons.

Indexing with negative values will access from the end of the array. For example,
the value negative one and negative two(-2) will return the last and second to
last element of the array, respectively. ArrayFire provides the `end` alias for
this which also allows you to index the last element of the array.

\snippet test/index.cpp index_tutorial_negative_indexing

## Indexing slices and subarrays

You can access regions of the array via the af::seq and af::span objects. The
span objects allows you to select the entire set of elements across a particular
dimension/axis of an array. For example, we can select the third column of the
array by passing span as the first argument and 2 as the second argument to the
parenthesis operator.

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

You can access the second row by passing (1, span) to the array

\snippet test/index.cpp index_tutorial_second_row

\f[ A(1, span) = [ 1, 5, 9, 13 ] \f]

You can use the af::seq (short for sequence) object to define a range when
indexing. For example, if you want to get the first two columns, you can access
the array by passing af::span for the first argument and af::seq(2) as the
second argument.

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

There are three constructors for af::seq.

* af::seq(N): Defines a range between 0 and N-1
* af::seq(begin, end) Defines a range between begin and end inclusive
* af::seq(begin, end, step) defines a range between begin and end striding by step values

The last constructor that can help create non-continuous ranges. For example,
you can select the second and forth(last) rows by passing (seq(1, end, 2), span)
to the indexing operator.

\snippet test/index.cpp index_tutorial_second_and_fourth_rows

\f[
A(seq(1, end, 2), span) =
\begin{bmatrix}
     1 & 5 &  9 & 13 \\
     3 & 7 & 11 & 15
\end{bmatrix}
\f]

## Indexing using af::array

You can also index using other af::array objects. ArrayFire performs a Cartesian
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

Boolean(b8) arrays can be used to index into another array. In this type of
indexing the non-zero values will be selected by the boolean operation. If we
want to select all values less than 5, we can pass a boolean expression into
the parenthesis operator.

\snippet test/index.cpp index_tutorial_boolean

\f[
out =
\begin{bmatrix}
0 \\ 1 \\ 2 \\ 3 \\ 4
\end{bmatrix}
\f]

## References and copies

All ArrayFire indexing functions return af::array(technically its an array_proxy
class) objects. These objects may be new arrays or they may reference the
original array depending on the type of indexing that was performed on them.

- If an array was indexed using another af::array or it was indexed using the
af::approx functions, then a new array is created. It does not reference the
original data.
- If an array was indexed using a scalar, af::seq or af::span, then
the resulting array will reference the original data IF the first dimension is
continuous. The following lines will not allocate additional memory.

\note The new arrays wither references or newly allocated arrays, are
independent of the original data. Meaning that any changes to the original array
will not propagate to the references. Likewise, any changes to the reference
arrays will not modify the original data.

\snippet test/index.cpp index_tutorial_references

The following code snippet shows some examples of indexing that will allocate
new memory.

\snippet test/index.cpp index_tutorial_copies

Notice that even though the copy3 array is referencing continuous memory in the
original array, a new array is created because we used an array to index into
the af::array.

## Assignment

An assignment on an af::array will replace the array with the result of the
expression on the right hand side of the equal(=) operator. This means that the
type and shape of the result can be different from the array on the left had
side of the equal operator. Assignments will not update the array that was
previously referenced through an indexing operation. Here is an example:

\snippet test/index.cpp index_tutorial_assignment

The `ref` array is created by indexing into the data array. The initialized
`ref` array points to the data array and does not allocate memory when it is
created. After the matmul call, the `ref` array will not be pointing to the data
array. The matmul call will not update the values of the data array.

You can update the contents of an af::array by assigning with the operator
parenthesis. For example, if you wanted to change the third column of the
`A` array you can do that by assigning to `A(span, 2)`.

\snippet test/index.cpp index_tutorial_assignment_third_column

\f[
ref =
\begin{bmatrix}
     8  \\
     9  \\
    10  \\
    11
\end{bmatrix}
A =
\begin{bmatrix}
    0 & 4 & 3.14 & 12 \\
    1 & 5 & 3.14 & 13 \\
    2 & 6 & 3.14 & 14 \\
    3 & 7 & 3.14 & 15
\end{bmatrix}
\f]

This will update only the array being modified. If there are arrays that
are referring to this array because of an indexing operation, those values
will remain unchanged.

Allocation will only be performed if there are other arrays referencing the data
at the point of assignment. In the previous example, an allocation will be
performed when assigning to the `A` array because the `ref` array is pointing
to the original data. Here is another example demonstrating when an allocation
will occur:

\snippet test/index.cpp index_tutorial_assignment_alloc

In this example, no allocation will take place because when the `ref` object
is created, it is pointing to `A`'s data. Once it goes out of scope, no data
points to `A`, therefore when the assignment takes place, the data is modified in
place instead of being copied to a new address.

You can also assign to arrays using another af::arrays as an indexing array.
This works in a similar way to the other types of assignment but care must be
taken to assure that the indexes are unique. Non-unique indexes will result in a
race condition which will cause non-deterministic values.

\snippet test/index.cpp index_tutorial_assignment_race_condition

\f[
idx =
\begin{bmatrix}
     4  \\
     3  \\
     4  \\
     0
\end{bmatrix}
vals =
\begin{bmatrix}
     9  \\
     8  \\
     7  \\
     6
\end{bmatrix}
\\
A =
\begin{bmatrix}
    6 & 9\ or\ 7 &  8 & 12 \\
    1 &   5    &  9 & 13 \\
    2 &   6    & 10 & 14 \\
    8 &   7    & 11 & 15
\end{bmatrix}
\f]

## Member functions

There are several member functions which allow you to index into an af::array. These
functions have similar functionality but may be easier to parse for some.

* [row(i)](\ref af::array::row) or [col(i)](\ref af::array::col) specifying a single row/column
* [rows(first,last)](\ref af::array::rows) or [cols(first,last)](\ref af::array::cols)
  specifying multiple rows or columns
* [slice(i)](\ref af::array::slice) or [slices(first, last)](\ref af::array::slices) to
  select one or a range of slices

# Additional examples

See \ref  index_mat for the full listing.

\snippet test/index.cpp ex_indexing_first

You can set values in an array:

\snippet test/index.cpp ex_indexing_set

Use one array to reference into another.

\snippet test/index.cpp ex_indexing_ref
