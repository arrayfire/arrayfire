Indexing {#indexing}
========

There are several ways of referencing values.  ArrayFire uses
parenthesis for subscripted referencing instead of the traditional
square bracket notation.  Indexing is zero-based, i.e. the first
element is at index zero (<tt>A(0)</tt>).  Indexing can be done
with mixtures of:
* integer scalars
* [seq()](\ref af::seq) representing a linear sequence
* [end](\ref af::end) representing the last element of a dimension
* [span](\ref af::span) representing the entire dimension
* [row(i)](\ref af::array::row) or [col(i)](\ref af::array::col) specifying a single row/column
* [rows(first,last)](\ref af::array::rows) or [cols(first,last)](\ref af::array::cols)
 specifying a span of rows or columns

See \ref gettingstarted_indexing for the full listing.

\democode{
array A = array(seq(1,9), 3, 3);
af_print(A);

af_print(A(0));    // first element
af_print(A(0,1));  // first row, second column

af_print(A(end));   // last element
af_print(A(-1));    // also last element
af_print(A(end-1)); // second-to-last element

af_print(A(1,span));       // second row
af_print(A.row(end));      // last row
af_print(A.cols(1,end));   // all but first column

float b_host[] = {0,1,2,3,4,5,6,7,8,9};
array b(10, 1, b_host);
af_print(b(seq(3)));
af_print(b(seq(1,7)));
af_print(b(seq(1,2,7)));
af_print(b(seq(0,2,end)));
}

You can set values in an array:

\democode{
array A = constant(0, 3, 3);

// setting entries to a constant
A(span) = 4;        // fill entire array
af_print(A);

A.row(0) = -1;      // first row
af_print(A);

A(seq(3)) = 3.1415; // first three elements
af_print(A);

// copy in another matrix
array B = constant(1, 4, 4, f64);
B.row(0) = randu(1, 4, f32); // set a row to random values (also upcast)
}


Use one array to reference into another.

\democode{
float h_inds[] = {0, 4, 2, 1}; // zero-based indexing
array inds(1, 4, h_inds);
af_print(inds);

array B = randu(1, 4);
af_print(B);

array c = B(inds);        // get
af_print(c);

B(inds) = -1;             // set to scalar
B(inds) = constant(0, 4); // zero indices
af_print(B);
}
