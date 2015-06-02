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

See \ref indexing for the full listing.

\snippet test/index.cpp ex_indexing_first

You can set values in an array:

\snippet test/index.cpp ex_indexing_set

Use one array to reference into another.

\snippet test/index.cpp ex_indexing_ref
