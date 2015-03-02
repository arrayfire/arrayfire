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
#include <af/data.h>

#include <vector>
#include <algorithm>
#include <functional>
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
    if (noDoubleTests<T>()) return;

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

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(a));
    for (size_t i = 0; i < indexed_array.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(indexed_array[i]));
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

        span_seqs.push_back(af_span);
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

        column_continuous_seq.push_back(make_vec(af_span, {  0,  6,  1}));
        column_continuous_seq.push_back(make_vec(af_span, {  4,  9,  1}));
        column_continuous_seq.push_back(make_vec(af_span, {  3,  8,  1}));

        column_continuous_reverse_seq.push_back(make_vec(af_span, {  6,  0,  -1}));
        column_continuous_reverse_seq.push_back(make_vec(af_span, {  9,  4,  -1}));
        column_continuous_reverse_seq.push_back(make_vec(af_span, {  8,  3,  -1}));

        column_strided_seq.push_back(make_vec(af_span, {  0,    8,   2 })); // Two Step
        column_strided_seq.push_back(make_vec(af_span, {  2,    9,   3 })); // Three Step
        column_strided_seq.push_back(make_vec(af_span, {  0,    9,   4 })); // Four Step

        column_strided_reverse_seq.push_back(make_vec(af_span, {  8,   0,   -2 })); // Two Step
        column_strided_reverse_seq.push_back(make_vec(af_span, {  9,   2,   -3 })); // Three Step
        column_strided_reverse_seq.push_back(make_vec(af_span, {  9,   0,   -4 })); // Four Step

        row_continuous_seq.push_back(make_vec({  0,  6,  1}, af_span));
        row_continuous_seq.push_back(make_vec({  4,  9,  1}, af_span));
        row_continuous_seq.push_back(make_vec({  3,  8,  1}, af_span));

        row_continuous_reverse_seq.push_back(make_vec({  6,  0,  -1}, af_span));
        row_continuous_reverse_seq.push_back(make_vec({  9,  4,  -1}, af_span));
        row_continuous_reverse_seq.push_back(make_vec({  8,  3,  -1}, af_span));

        row_strided_seq.push_back(make_vec({  0,    8,   2 }, af_span));
        row_strided_seq.push_back(make_vec({  2,    9,   3 }, af_span));
        row_strided_seq.push_back(make_vec({  0,    9,   4 }, af_span));

        row_strided_reverse_seq.push_back(make_vec({  8,   0,   -2 }, af_span));
        row_strided_reverse_seq.push_back(make_vec({  9,   2,   -3 }, af_span));
        row_strided_reverse_seq.push_back(make_vec({  9,   0,   -4 }, af_span));

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
    if (noDoubleTests<T>()) return;

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

    ASSERT_EQ(AF_SUCCESS, af_destroy_array(a));
    for (size_t i = 0; i < indexed_arrays.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(indexed_arrays[i]));
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


template<typename T>
class Indexing : public ::testing::Test
{
    vector<af_seq> make_vec3(af_seq first, af_seq second, af_seq third) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        out.push_back(third);
        return out;
    }

    vector<af_seq> make_vec4(af_seq first, af_seq second, af_seq third, af_seq fourth) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        out.push_back(third);
        out.push_back(fourth);
        return out;
    }

    public:

    virtual void SetUp() {
        continuous3d_to_3d.push_back(make_vec3({ 0, 4, 1}, { 0,  6,  1}, af_span));
        continuous3d_to_3d.push_back(make_vec3({ 4, 8, 1}, { 4,  9,  1}, af_span));
        continuous3d_to_3d.push_back(make_vec3({ 6, 9, 1}, { 3,  8,  1}, af_span));

        continuous3d_to_2d.push_back(make_vec3(af_span, { 0,  6,  1}, { 0, 0, 1}));
        continuous3d_to_2d.push_back(make_vec3(af_span, { 4,  9,  1}, { 1, 1, 1}));
        continuous3d_to_2d.push_back(make_vec3(af_span, { 3,  8,  1}, { 0, 0, 1}));

        continuous3d_to_1d.push_back(make_vec3(af_span, { 0,  0,  1}, { 0, 0, 1}));
        continuous3d_to_1d.push_back(make_vec3(af_span, { 6,  6,  1}, { 1, 1, 1}));
        continuous3d_to_1d.push_back(make_vec3(af_span, { 9,  9,  1}, { 0, 0, 1}));

        continuous4d_to_4d.push_back(make_vec4({ 2, 6, 1}, { 2,  6,  1}, af_span, af_span));
        continuous4d_to_3d.push_back(make_vec4({ 2, 6, 1}, { 2,  6,  1}, af_span, {0, 0, 1}));
        continuous4d_to_2d.push_back(make_vec4({ 2, 6, 1}, { 2,  6,  1}, { 0, 0, 1}, {0, 0, 1}));
        continuous4d_to_1d.push_back(make_vec4({ 2, 6, 1}, { 2,  2,  1}, { 0, 0, 1}, {0, 0, 1}));
    }

    vector<vector<af_seq>> continuous3d_to_3d;
    vector<vector<af_seq>> continuous3d_to_2d;
    vector<vector<af_seq>> continuous3d_to_1d;

    vector<vector<af_seq>> continuous4d_to_4d;
    vector<vector<af_seq>> continuous4d_to_3d;
    vector<vector<af_seq>> continuous4d_to_2d;
    vector<vector<af_seq>> continuous4d_to_1d;
};

template<typename T, size_t NDims>
void DimCheckND(const vector<vector<af_seq>> &seqs,string TestFile)
{
    if (noDoubleTests<T>()) return;

    // DimCheck2D function is generalized enough
    // to check 3d and 4d indexing
    DimCheck2D<T, NDims>(seqs, TestFile);
}

TYPED_TEST_CASE(Indexing, TestTypes);

TYPED_TEST(Indexing, 4D_to_4D)
{
    DimCheckND<TypeParam, 4>(this->continuous4d_to_4d, TEST_DIR"/index/Continuous4Dto4D.test");
}

TYPED_TEST(Indexing, 4D_to_3D)
{
    DimCheckND<TypeParam, 4>(this->continuous4d_to_3d, TEST_DIR"/index/Continuous4Dto3D.test");
}

TYPED_TEST(Indexing, 4D_to_2D)
{
    DimCheckND<TypeParam, 4>(this->continuous4d_to_2d, TEST_DIR"/index/Continuous4Dto2D.test");
}

TYPED_TEST(Indexing, 4D_to_1D)
{
    DimCheckND<TypeParam, 4>(this->continuous4d_to_1d, TEST_DIR"/index/Continuous4Dto1D.test");
}

TYPED_TEST(Indexing, 3D_to_3D)
{
    DimCheckND<TypeParam, 3>(this->continuous3d_to_3d, TEST_DIR"/index/Continuous3Dto3D.test");
}

TYPED_TEST(Indexing, 3D_to_2D)
{
    DimCheckND<TypeParam, 3>(this->continuous3d_to_2d, TEST_DIR"/index/Continuous3Dto2D.test");
}

TYPED_TEST(Indexing, 3D_to_1D)
{
    DimCheckND<TypeParam, 3>(this->continuous3d_to_1d, TEST_DIR"/index/Continuous3Dto1D.test");
}

//////////////////////////////// CPP ////////////////////////////////
TEST(Indexing2D, ColumnContiniousCPP)
{
    if (noDoubleTests<float>()) return;

    using af::array;

    vector<vector<af_seq>> seqs;

    seqs.push_back(make_vec(af_span, {  0,  6,  1}));
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

/************************ Array Based indexing tests from here on ******************/

template<typename T>
class ArrayIndex : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

typedef ::testing::Types<float, double, int, unsigned, unsigned char> ArrIdxTestTypes;
TYPED_TEST_CASE(ArrayIndex, ArrIdxTestTypes);

template<typename T>
void arrayIndexTest(string pTestFile, int dim)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>  numDims;
    vector<vector<T>>      in;
    vector<vector<T>>   tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array idxArray  = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                dims0.ndims(), dims0.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&idxArray, &(in[1].front()),
                dims1.ndims(), dims1.get(), (af_dtype)af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_SUCCESS, af_array_index(&outArray, inArray, idxArray, dim));

    vector<T> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    T *outData = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(idxArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TYPED_TEST(ArrayIndex, Dim0)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim0.test"), 0);
}

TYPED_TEST(ArrayIndex, Dim1)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim1.test"), 1);
}

TYPED_TEST(ArrayIndex, Dim2)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim2.test"), 2);
}

TYPED_TEST(ArrayIndex, Dim3)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim3.test"), 3);
}

TEST(ArrayIndex, CPP)
{
    using af::array;

    vector<af::dim4>      numDims;
    vector<vector<float>>      in;
    vector<vector<float>>   tests;

    readTests<float, float, int>(string(TEST_DIR"/arrayindex/dim0.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = input(indices);

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    output.host((void*)outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(SeqIndex, CPP_END)
{
    using af::array;

    const int n = 5;
    const int m = 5;
    const int end_off = 2;

    array a = af::randu(n, m);
    array b = a(af::end - end_off, af::span);

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < m; i++) {
        ASSERT_EQ(hA[i * n + end_off], hB[i]);
    }


    delete[] hA;
    delete[] hB;
}


TEST(SeqIndex, CPP_END_SEQ)
{
    using af::array;

    const int num = 20;
    const int end_begin = 10;
    const int end_end = 0;

    array a = af::randu(num);
    array b = a(af::seq(af::end - end_begin, af::end - end_end));

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < end_begin - end_end + 1; i++) {
        ASSERT_EQ(hA[i + end_begin - 1], hB[i]);
    }

    delete[] hA;
    delete[] hB;
}

af::array cpp_scope_test(const int num, const float val, const af::seq s)
{
    af::array a = af::constant(val, num);
    return a(s);
}

TEST(SeqIndex, CPP_SCOPE)
{
    using af::array;

    const int num = 20;
    const int seq_begin = 3;
    const int seq_end = 10;
    const float val = 133.33;

    array b = cpp_scope_test(num, val, af::seq(seq_begin, seq_end));
    float *hB = b.host<float>();

    for (int i = 0; i < seq_end - seq_begin + 1; i++) {
        ASSERT_EQ(hB[i], val);
    }

    delete[] hB;
}

TEST(SeqIndex, CPPLarge)
{
    using af::array;

    vector<af::dim4>      numDims;
    vector<vector<float>>      in;
    vector<vector<float>>   tests;

    readTests<float, float, int>(string(TEST_DIR"/arrayindex/dim0Large.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = input(indices);

    vector<float> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    float *outData = new float[nElems];

    output.host((void*)outData);

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
}

TEST(SeqIndex, Cascade00)
{
    using af::seq;
    using af::span;

    const int nx = 200;
    const int ny = 200;

    const int stb = 21;
    const int enb = 180;

    const int stc = 3;   // Should be less than nx - stb
    const int enc = 109; // Should be less than ny - enb

    const int st = stb + stc;
    const int en = stb + enc;
    const int nxc = en - st + 1;

    af::array a = af::randu(nx, ny);
    af::array b = a(seq(stb, enb), span);
    af::array c = b(seq(stc, enc), span);

    ASSERT_EQ(c.dims(1), ny );
    ASSERT_EQ(c.dims(0), nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = 0; j < ny; j++) {

        int a_off = j * nx;
        int c_off = j * nxc;

        for (int i = st; i < en; i++) {
            ASSERT_EQ(h_a[a_off + i],
                      h_c[c_off + i - st])
                << "at (" << i << "," << j << ")";
        }
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

TEST(SeqIndex, Cascade01)
{
    using af::seq;
    using af::span;

    const int nx = 200;
    const int ny = 200;

    const int stb = 54;
    const int enb = 196;

    const int stc = 39;
    const int enc = 123;

    const int nxc = enb - stb + 1;
    const int nyc = enc - stc + 1;

    af::array a = af::randu(nx, ny);
    af::array b = a(seq(stb, enb), span);
    af::array c = b(span, seq(stc, enc));

    ASSERT_EQ(c.dims(1), nyc);
    ASSERT_EQ(c.dims(0), nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = stc; j < enc; j++) {

        int a_off = j * nx;
        int c_off = (j - stc) * nxc;

        for (int i = stb; i < enb; i++) {

            ASSERT_EQ(h_a[a_off + i],
                      h_c[c_off + i - stb])
                << "at (" << i << "," << j << ")";
        }
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

TEST(SeqIndex, Cascade10)
{
    using af::seq;
    using af::span;

    const int nx = 200;
    const int ny = 200;

    const int stb = 71;
    const int enb = 188;

    const int stc = 33;
    const int enc = 155;

    const int nxc = enc - stc + 1;
    const int nyc = enb - stb + 1;

    af::array a = af::randu(nx, ny);
    af::array b = a(span, seq(stb, enb));
    af::array c = b(seq(stc, enc), span);

    ASSERT_EQ(c.dims(1), nyc);
    ASSERT_EQ(c.dims(0), nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = stb; j < enb; j++) {

        int a_off = j * nx;
        int c_off = (j - stb) * nxc;

        for (int i = stc; i < enc; i++) {

            ASSERT_EQ(h_a[a_off + i],
                      h_c[c_off + i - stc])
                << "at (" << i << "," << j << ")";
        }
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

TEST(SeqIndex, Cascade11)
{
    using af::seq;
    using af::span;

    const int nx = 200;
    const int ny = 200;

    const int stb = 50;
    const int enb = 150;

    const int stc = 20; // Should be less than nx - stb
    const int enc = 80; // Should be less than ny - enb

    const int st = stb + stc;
    const int en = stb + enc;
    const int nyc = en - st + 1;

    af::array a = af::randu(nx, ny);
    af::array b = a(span, seq(stb, enb));
    af::array c = b(span, seq(stc, enc));

    ASSERT_EQ(c.dims(1), nyc);
    ASSERT_EQ(c.dims(0), nx );

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = st; j < en; j++) {

        int a_off = j * nx;
        int c_off = (j - st) * nx;

        for (int i = 0; i < nx; i++) {

            ASSERT_EQ(h_a[a_off + i],
                      h_c[c_off + i])
                << "at (" << i << "," << j << ")";
        }
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}
