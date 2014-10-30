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
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::generate;
using std::cout;
using std::endl;
using std::ostream_iterator;
using af::dtype_traits;

template<typename T, typename OP>
void
checkValues(const af_seq &seq, const T* data, const T* indexed_data, OP compair_op) {
    for(int i = 0, j = seq.begin; compair_op(j,(int)seq.end); j+= seq.step, i++) {
        ASSERT_DOUBLE_EQ(data[j], indexed_data[i])
        << "Where i = " << i << " and j = " << j;
    }
}

template<typename T>
void
DimCheck(const vector<af_seq> &seqs) {
    static const int ndims = 1;
    static const size_t dims = 100;

    dim_type d[1] = {dims};

    vector<T> hData(dims);
    T n(0);
    generate(hData.begin(), hData.end(), [&] () { return n++; });

    af_array a = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&a, &hData.front(), ndims, d, (af_dtype) dtype_traits<T>::af_type));

    vector<af_array> indexed_array(seqs.size(), 0);
    for(size_t i = 0; i < seqs.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_index(&(indexed_array[i]), a, ndims, &seqs[i]))
            << "where seqs[i].begin == "    << seqs[i].begin
            << " seqs[i].step == "          << seqs[i].step
            << " seqs[i].end == "           << seqs[i].end;
    }

    vector<T*> h_indexed(seqs.size());
    for(size_t i = 0; i < seqs.size(); i++) {
        dim_type elems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&elems, indexed_array[i]));
        h_indexed[i] = new T[elems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void *)(h_indexed[i]), indexed_array[i]));
    }

    for(size_t k = 0; k < seqs.size(); k++) {
        if(seqs[k].step > 0)        {
            checkValues(seqs[k], &hData.front(), h_indexed[k], std::less_equal<int>());
        } else if (seqs[k].step < 0)  {
            checkValues(seqs[k], &hData.front(), h_indexed[k], std::greater_equal<int>());
        } else {
            for(size_t i = 0; i <= seqs[k].end; i++) {
                ASSERT_DOUBLE_EQ(hData[i], h_indexed[k][i])
                    << "Where i = " << i;
            }
        }
        delete[] h_indexed[k];
    }
}

template<typename T>
class Indexing1D : public ::testing::Test
{
public:
    virtual void SetUp() {
        continuous_seqs.push_back({  0,    20,   1 }); // Begin Continious
        continuous_seqs.push_back({  80,   99,   1 }); // End Continious
        continuous_seqs.push_back({  10,   89,   1 }); // Mid Continious

        continuous_reverse_seqs.push_back({  20,   0,    -1 }); // Begin Reverse Continious
        continuous_reverse_seqs.push_back({  99,   80,   -1 }); // End Reverse Continious
        continuous_reverse_seqs.push_back({  89,   10,   -1 }); // Mid Reverse Continious

        strided_seqs.push_back({  5,    40,   2 }); // Two Step
        strided_seqs.push_back({  5,    40,   3 }); // Three Step
        strided_seqs.push_back({  5,    40,   4 }); // Four Step

        strided_reverse_seqs.push_back({  40,    5,   -2 }); // Reverse Two Step
        strided_reverse_seqs.push_back({  40,    5,   -3 }); // Reverse Three Step
        strided_reverse_seqs.push_back({  40,    5,   -4 }); // Reverse Four Step

        span_seqs.push_back(span);
    }

    virtual ~Indexing1D() {}

    //virtual void TearDown() {}

    vector<af_seq> continuous_seqs;
    vector<af_seq> continuous_reverse_seqs;
    vector<af_seq> strided_seqs;
    vector<af_seq> strided_reverse_seqs;
    vector<af_seq> span_seqs;
};

typedef ::testing::Types<float, double, int, unsigned, char, unsigned char> TestTypes;
//typedef ::testing::Types<float> TestTypes;
TYPED_TEST_CASE(Indexing1D, TestTypes);

TYPED_TEST(Indexing1D, Continious)          { DimCheck<TypeParam>(this->continuous_seqs);           }
TYPED_TEST(Indexing1D, ContiniousReverse)   { DimCheck<TypeParam>(this->continuous_reverse_seqs);   }
TYPED_TEST(Indexing1D, Strided)             { DimCheck<TypeParam>(this->strided_seqs);              }
TYPED_TEST(Indexing1D, StridedReverse)      { DimCheck<TypeParam>(this->strided_reverse_seqs);      }
TYPED_TEST(Indexing1D, Span)                { DimCheck<TypeParam>(this->span_seqs);                 }


template<typename T>
class Indexing2D : public ::testing::Test
{
public:
    vector<af_seq> make_vec(af_seq first, af_seq second) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        return out;
    }
    virtual void SetUp() {

        column_continuous_seq.push_back(make_vec(span, {  0,  6,  1}));
        column_continuous_seq.push_back(make_vec(span, {  4,  9,  1}));
        column_continuous_seq.push_back(make_vec(span, {  3,  8,  1}));

        column_continuous_reverse_seq.push_back(make_vec(span, {  6,  0,  -1}));
        column_continuous_reverse_seq.push_back(make_vec(span, {  9,  4,  -1}));
        column_continuous_reverse_seq.push_back(make_vec(span, {  8,  3,  -1}));

        column_strided_seq.push_back(make_vec(span, {  0,    8,   2 })); // Two Step
        column_strided_seq.push_back(make_vec(span, {  2,    9,   3 })); // Three Step
        column_strided_seq.push_back(make_vec(span, {  0,    9,   4 })); // Four Step

        column_strided_reverse_seq.push_back(make_vec(span, {  8,   0,   -2 })); // Two Step
        column_strided_reverse_seq.push_back(make_vec(span, {  9,   2,   -3 })); // Three Step
        column_strided_reverse_seq.push_back(make_vec(span, {  9,   0,   -4 })); // Four Step

        row_continuous_seq.push_back(make_vec({  0,  6,  1}, span));
        row_continuous_seq.push_back(make_vec({  4,  9,  1}, span));
        row_continuous_seq.push_back(make_vec({  3,  8,  1}, span));

        row_continuous_reverse_seq.push_back(make_vec({  6,  0,  -1}, span));
        row_continuous_reverse_seq.push_back(make_vec({  9,  4,  -1}, span));
        row_continuous_reverse_seq.push_back(make_vec({  8,  3,  -1}, span));

        row_strided_seq.push_back(make_vec({  0,    8,   2 }, span));
        row_strided_seq.push_back(make_vec({  2,    9,   3 }, span));
        row_strided_seq.push_back(make_vec({  0,    9,   4 }, span));

        row_strided_reverse_seq.push_back(make_vec({  8,   0,   -2 }, span));
        row_strided_reverse_seq.push_back(make_vec({  9,   2,   -3 }, span));
        row_strided_reverse_seq.push_back(make_vec({  9,   0,   -4 }, span));

        continuous_continuous_seq.push_back(make_vec({  1,  6,  1}, {  0,  6,  1}));
        continuous_continuous_seq.push_back(make_vec({  3,  9,  1}, {  4,  9,  1}));
        continuous_continuous_seq.push_back(make_vec({  5,  8,  1}, {  3,  8,  1}));

        continuous_reverse_seq.push_back(make_vec({  1,  6,  1}, {  6,  0,  -1}));
        continuous_reverse_seq.push_back(make_vec({  3,  9,  1}, {  9,  4,  -1}));
        continuous_reverse_seq.push_back(make_vec({  5,  8,  1}, {  8,  3,  -1}));

        continuous_strided_seq.push_back(make_vec({  1,  6,  1}, {  0,  8,  2}));
        continuous_strided_seq.push_back(make_vec({  3,  9,  1}, {  2,  9,  3}));
        continuous_strided_seq.push_back(make_vec({  5,  8,  1}, {  1,  9,  4}));

        continuous_strided_reverse_seq.push_back(make_vec({  1,  6,  1}, {  8,  0,  -2}));
        continuous_strided_reverse_seq.push_back(make_vec({  3,  9,  1}, {  9,  2,  -3}));
        continuous_strided_reverse_seq.push_back(make_vec({  5,  8,  1}, {  9,  1,  -4}));

        reverse_continuous_seq.push_back(make_vec({  6,  1,  -1}, {  0,  6,  1}));
        reverse_continuous_seq.push_back(make_vec({  9,  3,  -1}, {  4,  9,  1}));
        reverse_continuous_seq.push_back(make_vec({  8,  5,  -1}, {  3,  8,  1}));

        reverse_reverse_seq.push_back(make_vec({  6,  1,  -1}, {  6,  0,  -1}));
        reverse_reverse_seq.push_back(make_vec({  9,  3,  -1}, {  9,  4,  -1}));
        reverse_reverse_seq.push_back(make_vec({  8,  5,  -1}, {  8,  3,  -1}));

        reverse_strided_seq.push_back(make_vec({  6,  1,  -1}, {  0,  8,  2}));
        reverse_strided_seq.push_back(make_vec({  9,  3,  -1}, {  2,  9,  3}));
        reverse_strided_seq.push_back(make_vec({  8,  5,  -1}, {  1,  9,  4}));

        reverse_strided_reverse_seq.push_back(make_vec({  6,  1,  -1}, {  8,  0,  -2}));
        reverse_strided_reverse_seq.push_back(make_vec({  9,  3,  -1}, {  9,  2,  -3}));
        reverse_strided_reverse_seq.push_back(make_vec({  8,  5,  -1}, {  9,  1,  -4}));

        strided_continuous_seq.push_back(make_vec({  0,  8,  2}, {  0,  6,  1}));
        strided_continuous_seq.push_back(make_vec({  2,  9,  3}, {  4,  9,  1}));
        strided_continuous_seq.push_back(make_vec({  1,  9,  4}, {  3,  8,  1}));

        strided_strided_seq.push_back(make_vec({  1,  6,  2}, {  0,  8,  2}));
        strided_strided_seq.push_back(make_vec({  3,  9,  2}, {  2,  9,  3}));
        strided_strided_seq.push_back(make_vec({  5,  8,  2}, {  1,  9,  4}));
        strided_strided_seq.push_back(make_vec({  1,  6,  3}, {  0,  8,  2}));
        strided_strided_seq.push_back(make_vec({  3,  9,  3}, {  2,  9,  3}));
        strided_strided_seq.push_back(make_vec({  5,  8,  3}, {  1,  9,  4}));
        strided_strided_seq.push_back(make_vec({  1,  6,  4}, {  0,  8,  2}));
        strided_strided_seq.push_back(make_vec({  3,  9,  4}, {  2,  9,  3}));
        strided_strided_seq.push_back(make_vec({  3,  8,  4}, {  1,  9,  4}));
        strided_strided_seq.push_back(make_vec({  3,  6,  4}, {  1,  9,  4}));
    }

    vector<vector<af_seq>> column_continuous_seq;
    vector<vector<af_seq>> column_continuous_reverse_seq;
    vector<vector<af_seq>> column_strided_seq;
    vector<vector<af_seq>> column_strided_reverse_seq;

    vector<vector<af_seq>> row_continuous_seq;
    vector<vector<af_seq>> row_continuous_reverse_seq;
    vector<vector<af_seq>> row_strided_seq;
    vector<vector<af_seq>> row_strided_reverse_seq;

    vector<vector<af_seq>> continuous_continuous_seq;
    vector<vector<af_seq>> continuous_strided_seq;
    vector<vector<af_seq>> continuous_reverse_seq;
    vector<vector<af_seq>> continuous_strided_reverse_seq;

    vector<vector<af_seq>> reverse_continuous_seq;
    vector<vector<af_seq>> reverse_reverse_seq;
    vector<vector<af_seq>> reverse_strided_seq;
    vector<vector<af_seq>> reverse_strided_reverse_seq;

    vector<vector<af_seq>> strided_continuous_seq;
    vector<vector<af_seq>> strided_strided_seq;
};

template<typename T, size_t NDims>
void
DimCheck2D(const vector<vector<af_seq>> &seqs,string TestFile)
{
    vector<af::dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);
    af::dim4 dimensions = numDims[0];

    af_array a = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&a, &(hData[0].front()), NDims, dimensions.get(), (af_dtype) af::dtype_traits<T>::af_type));

    vector<af_array> indexed_arrays(seqs.size(), 0);
    for(size_t i = 0; i < seqs.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_index(&(indexed_arrays[i]), a, NDims, seqs[i].data()));
    }

    vector<T*> h_indexed(seqs.size(), nullptr);
    for(size_t i = 0; i < seqs.size(); i++) {
        dim_type elems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&elems, indexed_arrays[i]));
        h_indexed[i] = new T[elems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void *)h_indexed[i], indexed_arrays[i]));

        T* ptr = h_indexed[i];
        if(false == equal(ptr, ptr + tests[i].size(), tests[i].begin())) {
            cout << "index data: ";
            copy(ptr, ptr + tests[i].size(), ostream_iterator<T>(cout, ", "));
            cout << endl << "file data: ";
            copy(tests[i].begin(), tests[i].end(), ostream_iterator<T>(cout, ", "));
            FAIL() << "indexed_array[" << i << "] FAILED" << endl;
        }
        delete[] h_indexed[i];
    }
}

TYPED_TEST_CASE(Indexing2D, TestTypes);

TYPED_TEST(Indexing2D, ColumnContinious)
{
    DimCheck2D<TypeParam, 2>(this->column_continuous_seq, TEST_DIR"/index/ColumnContinious.test");
}

TYPED_TEST(Indexing2D, ColumnContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->column_continuous_reverse_seq, TEST_DIR"/index/ColumnContiniousReverse.test");
}

TYPED_TEST(Indexing2D, ColumnStrided)
{
    DimCheck2D<TypeParam, 2>(this->column_strided_seq, TEST_DIR"/index/ColumnStrided.test");
}

TYPED_TEST(Indexing2D, ColumnStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->column_strided_reverse_seq, TEST_DIR"/index/ColumnStridedReverse.test");
}

TYPED_TEST(Indexing2D, RowContinious)
{
    DimCheck2D<TypeParam, 2>(this->row_continuous_seq, TEST_DIR"/index/RowContinious.test");
}

TYPED_TEST(Indexing2D, RowContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->row_continuous_reverse_seq, TEST_DIR"/index/RowContiniousReverse.test");
}

TYPED_TEST(Indexing2D, RowStrided)
{
    DimCheck2D<TypeParam, 2>(this->row_strided_seq, TEST_DIR"/index/RowStrided.test");
}

TYPED_TEST(Indexing2D, RowStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->row_strided_reverse_seq, TEST_DIR"/index/RowStridedReverse.test");
}

TYPED_TEST(Indexing2D, ContiniousContinious)
{
    DimCheck2D<TypeParam, 2>(this->continuous_continuous_seq, TEST_DIR"/index/ContiniousContinious.test");
}

TYPED_TEST(Indexing2D, ContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->continuous_reverse_seq, TEST_DIR"/index/ContiniousReverse.test");
}

TYPED_TEST(Indexing2D, ContiniousStrided)
{
    DimCheck2D<TypeParam, 2>(this->continuous_strided_seq, TEST_DIR"/index/ContiniousStrided.test");
}

TYPED_TEST(Indexing2D, ContiniousStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->continuous_strided_reverse_seq, TEST_DIR"/index/ContiniousStridedReverse.test");
}

TYPED_TEST(Indexing2D, ReverseContinious)
{
    DimCheck2D<TypeParam, 2>(this->reverse_continuous_seq, TEST_DIR"/index/ReverseContinious.test");
}

TYPED_TEST(Indexing2D, ReverseReverse)
{
    DimCheck2D<TypeParam, 2>(this->reverse_reverse_seq, TEST_DIR"/index/ReverseReverse.test");
}

TYPED_TEST(Indexing2D, ReverseStrided)
{
    DimCheck2D<TypeParam, 2>(this->reverse_strided_seq, TEST_DIR"/index/ReverseStrided.test");
}

TYPED_TEST(Indexing2D, ReverseStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->reverse_strided_reverse_seq, TEST_DIR"/index/ReverseStridedReverse.test");
}

TYPED_TEST(Indexing2D, StridedContinious)
{
    DimCheck2D<TypeParam, 2>(this->strided_continuous_seq, TEST_DIR"/index/StridedContinious.test");
}

TYPED_TEST(Indexing2D, StridedStrided)
{
    DimCheck2D<TypeParam, 2>(this->strided_strided_seq, TEST_DIR"/index/StridedStrided.test");
}

vector<af_seq> make_vec(af_seq first, af_seq second) {
    vector<af_seq> out;
    out.push_back(first);
    out.push_back(second);
    return out;
}

TEST(Indexing2D, ColumnContiniousCPP)
{
    using af::array;

    vector<vector<af_seq>> seqs;

    seqs.push_back(make_vec(span, {  0,  6,  1}));
    //seqs.push_back(make_vec(span, {  4,  9,  1}));
    //seqs.push_back(make_vec(span, {  3,  8,  1}));

    vector<af::dim4> numDims;

    vector<vector<float>> hData;
    vector<vector<float>> tests;
    readTests<float, float, int>(TEST_DIR"/index/ColumnContinious.test", numDims, hData, tests);
    af::dim4 dimensions = numDims[0];

    array a(dimensions,&(hData[0].front()));

    vector<array> sub;
    for(size_t i = 0; i < seqs.size(); i++) {
        vector<af_seq> seq = seqs[i];
        sub.emplace_back(a(seq[0], seq[1]));
    }

    for(size_t i = 0; i < seqs.size(); i++) {
        dim_type elems = sub[i].elements();
        float *ptr = new float[elems];
        sub[i].host(ptr);

        if(false == equal(ptr, ptr + tests[i].size(), tests[i].begin())) {
            cout << "index data: ";
            copy(ptr, ptr + tests[i].size(), ostream_iterator<float>(cout, ", "));
            cout << endl << "file data: ";
            copy(tests[i].begin(), tests[i].end(), ostream_iterator<float>(cout, ", "));
            FAIL() << "indexed_array[" << i << "] FAILED" << endl;
        }
        delete[] ptr;
    }
}
