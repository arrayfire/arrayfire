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
        ASSERT_DOUBLE_EQ(real(data[j]), real(indexed_data[i]))
        << "Where i = " << i << " and j = " << j;
    }
}

template<typename T>
void
DimCheck(const vector<af_seq> &seqs) {
    if (noDoubleTests<T>()) return;

    static const int ndims = 1;
    static const size_t dims = 100;

    dim_t d[1] = {dims};

    vector<T> hData(dims);
    for(int i = 0; i < (int)dims; i++) { hData[i] = i; }

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
        dim_t elems;
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
                ASSERT_DOUBLE_EQ(real(hData[i]), real(h_indexed[k][i]))
                    << "Where i = " << i;
            }
        }
        delete[] h_indexed[k];
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(a));
    for (size_t i = 0; i < indexed_array.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_release_array(indexed_array[i]));
    }
}

template<typename T>
class Indexing1D : public ::testing::Test
{
public:
    virtual void SetUp() {
        continuous_seqs.push_back(af_make_seq(  0,    20,   1 )); // Begin Continious
        continuous_seqs.push_back(af_make_seq(  80,   99,   1 )); // End Continious
        continuous_seqs.push_back(af_make_seq(  10,   89,   1 )); // Mid Continious

        continuous_reverse_seqs.push_back(af_make_seq(  20,   0,    -1 )); // Begin Reverse Continious
        continuous_reverse_seqs.push_back(af_make_seq(  99,   80,   -1 )); // End Reverse Continious
        continuous_reverse_seqs.push_back(af_make_seq(  89,   10,   -1 )); // Mid Reverse Continious

        strided_seqs.push_back(af_make_seq(  5,    40,   2 )); // Two Step
        strided_seqs.push_back(af_make_seq(  5,    40,   3 )); // Three Step
        strided_seqs.push_back(af_make_seq(  5,    40,   4 )); // Four Step

        strided_reverse_seqs.push_back(af_make_seq(  40,    5,   -2 )); // Reverse Two Step
        strided_reverse_seqs.push_back(af_make_seq(  40,    5,   -3 )); // Reverse Three Step
        strided_reverse_seqs.push_back(af_make_seq(  40,    5,   -4 )); // Reverse Four Step

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

typedef ::testing::Types<float, double, af::cfloat, af::cdouble, int, unsigned, unsigned char, intl, uintl> AllTypes;
TYPED_TEST_CASE(Indexing1D, AllTypes);

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

        column_continuous_seq.push_back(make_vec(af_span, af_make_seq(  0,  6,  1)));
        column_continuous_seq.push_back(make_vec(af_span, af_make_seq(  4,  9,  1)));
        column_continuous_seq.push_back(make_vec(af_span, af_make_seq(  3,  8,  1)));

        column_continuous_reverse_seq.push_back(make_vec(af_span, af_make_seq(  6,  0,  -1)));
        column_continuous_reverse_seq.push_back(make_vec(af_span, af_make_seq(  9,  4,  -1)));
        column_continuous_reverse_seq.push_back(make_vec(af_span, af_make_seq(  8,  3,  -1)));

        column_strided_seq.push_back(make_vec(af_span, af_make_seq(  0,    8,   2 ))); // Two Step
        column_strided_seq.push_back(make_vec(af_span, af_make_seq(  2,    9,   3 ))); // Three Step
        column_strided_seq.push_back(make_vec(af_span, af_make_seq(  0,    9,   4 ))); // Four Step

        column_strided_reverse_seq.push_back(make_vec(af_span, af_make_seq(  8,   0,   -2 ))); // Two Step
        column_strided_reverse_seq.push_back(make_vec(af_span, af_make_seq(  9,   2,   -3 ))); // Three Step
        column_strided_reverse_seq.push_back(make_vec(af_span, af_make_seq(  9,   0,   -4 ))); // Four Step

        row_continuous_seq.push_back(make_vec(af_make_seq(  0,  6,  1), af_span));
        row_continuous_seq.push_back(make_vec(af_make_seq(  4,  9,  1), af_span));
        row_continuous_seq.push_back(make_vec(af_make_seq(  3,  8,  1), af_span));

        row_continuous_reverse_seq.push_back(make_vec(af_make_seq(  6,  0,  -1), af_span));
        row_continuous_reverse_seq.push_back(make_vec(af_make_seq(  9,  4,  -1), af_span));
        row_continuous_reverse_seq.push_back(make_vec(af_make_seq(  8,  3,  -1), af_span));

        row_strided_seq.push_back(make_vec(af_make_seq(  0,    8,   2 ), af_span));
        row_strided_seq.push_back(make_vec(af_make_seq(  2,    9,   3 ), af_span));
        row_strided_seq.push_back(make_vec(af_make_seq(  0,    9,   4 ), af_span));

        row_strided_reverse_seq.push_back(make_vec(af_make_seq(  8,   0,   -2 ), af_span));
        row_strided_reverse_seq.push_back(make_vec(af_make_seq(  9,   2,   -3 ), af_span));
        row_strided_reverse_seq.push_back(make_vec(af_make_seq(  9,   0,   -4 ), af_span));

        continuous_continuous_seq.push_back(make_vec(af_make_seq(  1,  6,  1), af_make_seq(  0,  6,  1)));
        continuous_continuous_seq.push_back(make_vec(af_make_seq(  3,  9,  1), af_make_seq(  4,  9,  1)));
        continuous_continuous_seq.push_back(make_vec(af_make_seq(  5,  8,  1), af_make_seq(  3,  8,  1)));

        continuous_reverse_seq.push_back(make_vec(af_make_seq(  1,  6,  1), af_make_seq(  6,  0,  -1)));
        continuous_reverse_seq.push_back(make_vec(af_make_seq(  3,  9,  1), af_make_seq(  9,  4,  -1)));
        continuous_reverse_seq.push_back(make_vec(af_make_seq(  5,  8,  1), af_make_seq(  8,  3,  -1)));

        continuous_strided_seq.push_back(make_vec(af_make_seq(  1,  6,  1), af_make_seq(  0,  8,  2)));
        continuous_strided_seq.push_back(make_vec(af_make_seq(  3,  9,  1), af_make_seq(  2,  9,  3)));
        continuous_strided_seq.push_back(make_vec(af_make_seq(  5,  8,  1), af_make_seq(  1,  9,  4)));

        continuous_strided_reverse_seq.push_back(make_vec(af_make_seq(  1,  6,  1), af_make_seq(  8,  0,  -2)));
        continuous_strided_reverse_seq.push_back(make_vec(af_make_seq(  3,  9,  1), af_make_seq(  9,  2,  -3)));
        continuous_strided_reverse_seq.push_back(make_vec(af_make_seq(  5,  8,  1), af_make_seq(  9,  1,  -4)));

        reverse_continuous_seq.push_back(make_vec(af_make_seq(  6,  1,  -1), af_make_seq(  0,  6,  1)));
        reverse_continuous_seq.push_back(make_vec(af_make_seq(  9,  3,  -1), af_make_seq(  4,  9,  1)));
        reverse_continuous_seq.push_back(make_vec(af_make_seq(  8,  5,  -1), af_make_seq(  3,  8,  1)));

        reverse_reverse_seq.push_back(make_vec(af_make_seq(  6,  1,  -1), af_make_seq(  6,  0,  -1)));
        reverse_reverse_seq.push_back(make_vec(af_make_seq(  9,  3,  -1), af_make_seq(  9,  4,  -1)));
        reverse_reverse_seq.push_back(make_vec(af_make_seq(  8,  5,  -1), af_make_seq(  8,  3,  -1)));

        reverse_strided_seq.push_back(make_vec(af_make_seq(  6,  1,  -1), af_make_seq(  0,  8,  2)));
        reverse_strided_seq.push_back(make_vec(af_make_seq(  9,  3,  -1), af_make_seq(  2,  9,  3)));
        reverse_strided_seq.push_back(make_vec(af_make_seq(  8,  5,  -1), af_make_seq(  1,  9,  4)));

        reverse_strided_reverse_seq.push_back(make_vec(af_make_seq(  6,  1,  -1), af_make_seq(  8,  0,  -2)));
        reverse_strided_reverse_seq.push_back(make_vec(af_make_seq(  9,  3,  -1), af_make_seq(  9,  2,  -3)));
        reverse_strided_reverse_seq.push_back(make_vec(af_make_seq(  8,  5,  -1), af_make_seq(  9,  1,  -4)));

        strided_continuous_seq.push_back(make_vec(af_make_seq(  0,  8,  2), af_make_seq(  0,  6,  1)));
        strided_continuous_seq.push_back(make_vec(af_make_seq(  2,  9,  3), af_make_seq(  4,  9,  1)));
        strided_continuous_seq.push_back(make_vec(af_make_seq(  1,  9,  4), af_make_seq(  3,  8,  1)));

        strided_strided_seq.push_back(make_vec(af_make_seq(  1,  6,  2), af_make_seq(  0,  8,  2)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  3,  9,  2), af_make_seq(  2,  9,  3)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  5,  8,  2), af_make_seq(  1,  9,  4)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  1,  6,  3), af_make_seq(  0,  8,  2)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  3,  9,  3), af_make_seq(  2,  9,  3)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  5,  8,  3), af_make_seq(  1,  9,  4)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  1,  6,  4), af_make_seq(  0,  8,  2)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  3,  9,  4), af_make_seq(  2,  9,  3)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  3,  8,  4), af_make_seq(  1,  9,  4)));
        strided_strided_seq.push_back(make_vec(af_make_seq(  3,  6,  4), af_make_seq(  1,  9,  4)));
    }

    vector<vector<af_seq> > column_continuous_seq;
    vector<vector<af_seq> > column_continuous_reverse_seq;
    vector<vector<af_seq> > column_strided_seq;
    vector<vector<af_seq> > column_strided_reverse_seq;

    vector<vector<af_seq> > row_continuous_seq;
    vector<vector<af_seq> > row_continuous_reverse_seq;
    vector<vector<af_seq> > row_strided_seq;
    vector<vector<af_seq> > row_strided_reverse_seq;

    vector<vector<af_seq> > continuous_continuous_seq;
    vector<vector<af_seq> > continuous_strided_seq;
    vector<vector<af_seq> > continuous_reverse_seq;
    vector<vector<af_seq> > continuous_strided_reverse_seq;

    vector<vector<af_seq> > reverse_continuous_seq;
    vector<vector<af_seq> > reverse_reverse_seq;
    vector<vector<af_seq> > reverse_strided_seq;
    vector<vector<af_seq> > reverse_strided_reverse_seq;

    vector<vector<af_seq> > strided_continuous_seq;
    vector<vector<af_seq> > strided_strided_seq;
};

template<typename T>
void
DimCheck2D(const vector<vector<af_seq> > &seqs,string TestFile, size_t NDims)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;

    vector<vector<T> > hData;
    vector<vector<T> > tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);
    af::dim4 dimensions = numDims[0];

    af_array a = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&a, &(hData[0].front()), NDims, dimensions.get(), (af_dtype) af::dtype_traits<T>::af_type));

    vector<af_array> indexed_arrays(seqs.size(), 0);
    for(size_t i = 0; i < seqs.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_index(&(indexed_arrays[i]), a, NDims, seqs[i].data()));
    }

    vector<T*> h_indexed(seqs.size(), NULL);
    for(size_t i = 0; i < seqs.size(); i++) {
        dim_t elems;
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

    ASSERT_EQ(AF_SUCCESS, af_release_array(a));
    for (size_t i = 0; i < indexed_arrays.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_release_array(indexed_arrays[i]));
    }
}

TYPED_TEST_CASE(Indexing2D, AllTypes);

TYPED_TEST(Indexing2D, ColumnContinious)
{
    DimCheck2D<TypeParam>(this->column_continuous_seq, TEST_DIR"/index/ColumnContinious.test", 2);
}

TYPED_TEST(Indexing2D, ColumnContiniousReverse)
{
    DimCheck2D<TypeParam>(this->column_continuous_reverse_seq, TEST_DIR"/index/ColumnContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, ColumnStrided)
{
    DimCheck2D<TypeParam>(this->column_strided_seq, TEST_DIR"/index/ColumnStrided.test", 2);
}

TYPED_TEST(Indexing2D, ColumnStridedReverse)
{
    DimCheck2D<TypeParam>(this->column_strided_reverse_seq, TEST_DIR"/index/ColumnStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, RowContinious)
{
    DimCheck2D<TypeParam>(this->row_continuous_seq, TEST_DIR"/index/RowContinious.test", 2);
}

TYPED_TEST(Indexing2D, RowContiniousReverse)
{
    DimCheck2D<TypeParam>(this->row_continuous_reverse_seq, TEST_DIR"/index/RowContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, RowStrided)
{
    DimCheck2D<TypeParam>(this->row_strided_seq, TEST_DIR"/index/RowStrided.test", 2);
}

TYPED_TEST(Indexing2D, RowStridedReverse)
{
    DimCheck2D<TypeParam>(this->row_strided_reverse_seq, TEST_DIR"/index/RowStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousContinious)
{
    DimCheck2D<TypeParam>(this->continuous_continuous_seq, TEST_DIR"/index/ContiniousContinious.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousReverse)
{
    DimCheck2D<TypeParam>(this->continuous_reverse_seq, TEST_DIR"/index/ContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousStrided)
{
    DimCheck2D<TypeParam>(this->continuous_strided_seq, TEST_DIR"/index/ContiniousStrided.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousStridedReverse)
{
    DimCheck2D<TypeParam>(this->continuous_strided_reverse_seq, TEST_DIR"/index/ContiniousStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, ReverseContinious)
{
    DimCheck2D<TypeParam>(this->reverse_continuous_seq, TEST_DIR"/index/ReverseContinious.test", 2);
}

TYPED_TEST(Indexing2D, ReverseReverse)
{
    DimCheck2D<TypeParam>(this->reverse_reverse_seq, TEST_DIR"/index/ReverseReverse.test", 2);
}

TYPED_TEST(Indexing2D, ReverseStrided)
{
    DimCheck2D<TypeParam>(this->reverse_strided_seq, TEST_DIR"/index/ReverseStrided.test", 2);
}

TYPED_TEST(Indexing2D, ReverseStridedReverse)
{
    DimCheck2D<TypeParam>(this->reverse_strided_reverse_seq, TEST_DIR"/index/ReverseStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, StridedContinious)
{
    DimCheck2D<TypeParam>(this->strided_continuous_seq, TEST_DIR"/index/StridedContinious.test", 2);
}

TYPED_TEST(Indexing2D, StridedStrided)
{
    DimCheck2D<TypeParam>(this->strided_strided_seq, TEST_DIR"/index/StridedStrided.test", 2);
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
        continuous3d_to_3d.push_back(make_vec3(af_make_seq( 0, 4, 1), af_make_seq( 0,  6,  1), af_span));
        continuous3d_to_3d.push_back(make_vec3(af_make_seq( 4, 8, 1), af_make_seq( 4,  9,  1), af_span));
        continuous3d_to_3d.push_back(make_vec3(af_make_seq( 6, 9, 1), af_make_seq( 3,  8,  1), af_span));

        continuous3d_to_2d.push_back(make_vec3(af_span, af_make_seq( 0,  6,  1), af_make_seq( 0, 0, 1)));
        continuous3d_to_2d.push_back(make_vec3(af_span, af_make_seq( 4,  9,  1), af_make_seq( 1, 1, 1)));
        continuous3d_to_2d.push_back(make_vec3(af_span, af_make_seq( 3,  8,  1), af_make_seq( 0, 0, 1)));

        continuous3d_to_1d.push_back(make_vec3(af_span, af_make_seq( 0,  0,  1), af_make_seq( 0, 0, 1)));
        continuous3d_to_1d.push_back(make_vec3(af_span, af_make_seq( 6,  6,  1), af_make_seq( 1, 1, 1)));
        continuous3d_to_1d.push_back(make_vec3(af_span, af_make_seq( 9,  9,  1), af_make_seq( 0, 0, 1)));

        continuous4d_to_4d.push_back(make_vec4(af_make_seq( 2, 6, 1), af_make_seq( 2,  6,  1), af_span, af_span));
        continuous4d_to_3d.push_back(make_vec4(af_make_seq( 2, 6, 1), af_make_seq( 2,  6,  1), af_span, af_make_seq(0, 0, 1)));
        continuous4d_to_2d.push_back(make_vec4(af_make_seq( 2, 6, 1), af_make_seq( 2,  6,  1), af_make_seq( 0, 0, 1), af_make_seq(0, 0, 1)));
        continuous4d_to_1d.push_back(make_vec4(af_make_seq( 2, 6, 1), af_make_seq( 2,  2,  1), af_make_seq( 0, 0, 1), af_make_seq(0, 0, 1)));
    }

    vector<vector<af_seq> > continuous3d_to_3d;
    vector<vector<af_seq> > continuous3d_to_2d;
    vector<vector<af_seq> > continuous3d_to_1d;

    vector<vector<af_seq> > continuous4d_to_4d;
    vector<vector<af_seq> > continuous4d_to_3d;
    vector<vector<af_seq> > continuous4d_to_2d;
    vector<vector<af_seq> > continuous4d_to_1d;
};

template<typename T>
void DimCheckND(const vector<vector<af_seq> > &seqs,string TestFile, size_t NDims)
{
    if (noDoubleTests<T>()) return;

    // DimCheck2D function is generalized enough
    // to check 3d and 4d indexing
    DimCheck2D<T>(seqs, TestFile, NDims);
}

TYPED_TEST_CASE(Indexing, AllTypes);

TYPED_TEST(Indexing, 4D_to_4D)
{
    DimCheckND<TypeParam>(this->continuous4d_to_4d, TEST_DIR"/index/Continuous4Dto4D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_3D)
{
    DimCheckND<TypeParam>(this->continuous4d_to_3d, TEST_DIR"/index/Continuous4Dto3D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_2D)
{
    DimCheckND<TypeParam>(this->continuous4d_to_2d, TEST_DIR"/index/Continuous4Dto2D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_1D)
{
    DimCheckND<TypeParam>(this->continuous4d_to_1d, TEST_DIR"/index/Continuous4Dto1D.test", 4);
}

TYPED_TEST(Indexing, 3D_to_3D)
{
    DimCheckND<TypeParam>(this->continuous3d_to_3d, TEST_DIR"/index/Continuous3Dto3D.test", 3);
}

TYPED_TEST(Indexing, 3D_to_2D)
{
    DimCheckND<TypeParam>(this->continuous3d_to_2d, TEST_DIR"/index/Continuous3Dto2D.test", 3);
}

TYPED_TEST(Indexing, 3D_to_1D)
{
    DimCheckND<TypeParam>(this->continuous3d_to_1d, TEST_DIR"/index/Continuous3Dto1D.test", 3);
}

//////////////////////////////// CPP ////////////////////////////////
TEST(Indexing2D, ColumnContiniousCPP)
{
    if (noDoubleTests<float>()) return;

    using af::array;

    vector<vector<af_seq> > seqs;

    seqs.push_back(make_vec(af_span, af_make_seq(  0,  6,  1)));
    //seqs.push_back(make_vec(span, af_make_seq(  4,  9,  1)));
    //seqs.push_back(make_vec(span, af_make_seq(  3,  8,  1)));

    vector<af::dim4> numDims;

    vector<vector<float> > hData;
    vector<vector<float> > tests;
    readTests<float, float, int>(TEST_DIR"/index/ColumnContinious.test", numDims, hData, tests);
    af::dim4 dimensions = numDims[0];

    array a(dimensions,&(hData[0].front()));

    vector<array> sub;
    for(size_t i = 0; i < seqs.size(); i++) {
        vector<af_seq> seq = seqs[i];
        sub.push_back(a(seq[0], seq[1]));
    }

    for(size_t i = 0; i < seqs.size(); i++) {
        dim_t elems = sub[i].elements();
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
class lookup : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

typedef ::testing::Types<float, double, int, unsigned, unsigned char> ArrIdxTestTypes;
TYPED_TEST_CASE(lookup, ArrIdxTestTypes);

template<typename T>
void arrayIndexTest(string pTestFile, int dim)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4>  numDims;
    vector<vector<T> >      in;
    vector<vector<T> >   tests;

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

    ASSERT_EQ(AF_SUCCESS, af_lookup(&outArray, inArray, idxArray, dim));

    vector<T> currGoldBar = tests[0];
    size_t nElems = currGoldBar.size();
    T *outData = new T[nElems];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t elIter=0; elIter<nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])<< "at: " << elIter<< std::endl;
    }

    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idxArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(lookup, Dim0)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim0.test"), 0);
}

TYPED_TEST(lookup, Dim1)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim1.test"), 1);
}

TYPED_TEST(lookup, Dim2)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim2.test"), 2);
}

TYPED_TEST(lookup, Dim3)
{
    arrayIndexTest<TypeParam>(string(TEST_DIR"/arrayindex/dim3.test"), 3);
}

TEST(lookup, CPP)
{
    using af::array;

    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/arrayindex/dim0.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = af::lookup(input, indices, 0);

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

af::array cpp_scope_seq_test(const int num, const float val, const af::seq s)
{
    af::array a = af::constant(val, num);
    return a(s);
}

TEST(SeqIndex, CPP_SCOPE_SEQ)
{
    using af::array;

    const int num = 20;
    const int seq_begin = 3;
    const int seq_end = 10;
    const float val = 133.33;

    array b = cpp_scope_seq_test(num, val, af::seq(seq_begin, seq_end));
    float *hB = b.host<float>();

    for (int i = 0; i < seq_end - seq_begin + 1; i++) {
        ASSERT_EQ(hB[i], val);
    }

    delete[] hB;
}

af::array cpp_scope_arr_test(const int num, const float val)
{
    af::array a = af::constant(val, num);
    af::array idx = where(a > val/2);
    return a(idx) * (val - 1);
}

TEST(SeqIndex, CPP_SCOPE_ARR)
{
    using af::array;

    const int num = 20;
    const float val = 133.33;

    array b = cpp_scope_arr_test(num, val);
    float *hB = b.host<float>();

    for (int i = 0; i < (int)b.elements(); i++) {
        ASSERT_EQ(hB[i], val * (val - 1));
    }

    delete[] hB;
}

TEST(SeqIndex, CPPLarge)
{
    using af::array;

    vector<af::dim4>      numDims;
    vector<vector<float> >      in;
    vector<vector<float> >   tests;

    readTests<float, float, int>(string(TEST_DIR"/arrayindex/dim0Large.test"), numDims, in, tests);

    af::dim4 dims0     = numDims[0];
    af::dim4 dims1     = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = af::lookup(input, indices, 0);

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

    ASSERT_EQ(c.dims(1), (dim_t)ny );
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

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

    ASSERT_EQ(c.dims(1), (dim_t)nyc);
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

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

    ASSERT_EQ(c.dims(1), (dim_t)nyc);
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

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

TEST(ArrayIndex, CPP_INDEX_VECTOR)
{
    using af::array;
    float h_inds[] = {0, 3, 2, 1}; // zero-based indexing
    array inds(1, 4, h_inds);
    array B = af::randu(1, 4);
    array C = B(inds);

    ASSERT_EQ(B.dims(0), 1);
    ASSERT_EQ(B.dims(1), 4);
    ASSERT_EQ(C.dims(0), 1);
    ASSERT_EQ(C.dims(1), 4);

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(h_C[i], h_B[(int)h_inds[i]]);
    }

    delete[] h_B;
    delete[] h_C;
}

TEST(SeqIndex, CPP_INDEX_VECTOR)
{
    using af::array;

    const int num = 20;
    const int len = 10;
    const int st  =  3;
    const int en  = st + len - 1;

    array B = af::randu(1, 20);
    array C = B(af::seq(st, en));

    ASSERT_EQ(1  , B.dims(0));
    ASSERT_EQ(num, B.dims(1));
    ASSERT_EQ(1  , C.dims(0));
    ASSERT_EQ(len, C.dims(1));

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < len; i++) {
        ASSERT_EQ(h_C[i], h_B[i + st]);
    }

    delete[] h_B;
    delete[] h_C;
}


TEST(ArrayIndex, CPP_INDEX_VECTOR_2D)
{
    using af::array;
    float h_inds[] = {3, 5, 7, 2}; // zero-based indexing
    array inds(1, 4, h_inds);
    array B = af::randu(4, 4);
    array C = B(inds);

    ASSERT_EQ(B.dims(0), 4);
    ASSERT_EQ(B.dims(1), 4);
    ASSERT_EQ(C.dims(0), 4);
    ASSERT_EQ(C.dims(1), 1);

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(h_C[i], h_B[(int)h_inds[i]]);
    }

    delete[] h_B;
    delete[] h_C;
}

TEST(SeqIndex, CPP_INDEX_VECTOR_2D)
{
    using af::array;

    const int nx = 4;
    const int ny = 3 * nx;
    const int len = 2 * nx;
    const int st  = nx - 1;
    const int en  = st + len - 1;

    array B = af::randu(nx, ny);
    array C = B(af::seq(st, en));

    ASSERT_EQ(nx , B.dims(0));
    ASSERT_EQ(ny , B.dims(1));
    ASSERT_EQ(len, C.dims(0));
    ASSERT_EQ(1  , C.dims(1));

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < len; i++) {
        ASSERT_EQ(h_C[i], h_B[i + st]);
    }

    delete[] h_B;
    delete[] h_C;
}

template<typename T>
class IndexedMembers : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

TYPED_TEST_CASE(IndexedMembers, AllTypes);

TYPED_TEST(IndexedMembers, MemFuncs)
{
    if (noDoubleTests<TypeParam>()) return;
    using af::array;
    dim_t dimsize = 100;
    vector<TypeParam> in(dimsize * dimsize);
    for(int i = 0; i < (int)in.size(); i++) in[i] = i;
    array input(dimsize, dimsize, &in.front(), afHost);

    ASSERT_EQ(dimsize, input(af::span, 1).elements());
    ASSERT_EQ(input.type(), input(af::span, 1).type());
    ASSERT_EQ(af::dim4(dimsize), input(af::span, 1).dims());
    ASSERT_EQ(1u, input(af::span, 1).numdims());
    ASSERT_FALSE(input(af::span, 1).isempty());
    ASSERT_FALSE(input(af::span, 1).isscalar());
    ASSERT_TRUE(input(1, 1).isscalar());
    ASSERT_TRUE(input(af::span, 1).isvector());
    ASSERT_FALSE(input(af::span, 1).isrow());
    ASSERT_EQ(input.iscomplex(), input(af::span, 1).iscomplex());
    ASSERT_EQ(input.isdouble(), input(af::span, 1).isdouble());
    ASSERT_EQ(input.issingle(), input(af::span, 1).issingle());
    ASSERT_EQ(input.isrealfloating(), input(af::span, 1).isrealfloating());
    ASSERT_EQ(input.isfloating(), input(af::span, 1).isfloating());
    ASSERT_EQ(input.isinteger(), input(af::span, 1).isinteger());
    ASSERT_EQ(input.isbool(), input(af::span, 1).isbool());
    // TODO: Doesn't compile in cuda for cfloat and cdouble
    //ASSERT_EQ(input.scalar<TypeParam>(), input(af::span, 0).scalar<TypeParam>());
}


#if 0
TYPED_TEST(IndexedMembers, MemIndex)
{
    using namespace af;
    array a = range(dim4(10, 10));
    array b = a(seq(1,7), span);
    array brow = b.row(5);
    array brows = b.rows(5, 6);
    array bcol = b.col(5);
    array bcols = b.cols(5, 6);

    array out_row = a(seq(1,7), span).row(5);
    array out_rows = a(seq(1,7), span).rows(5, 6);
    array out_col = a(seq(1,7), span).col(5);
    array out_cols = a(seq(1,7), span).cols(5, 6);

    ASSERT_EQ(0, where(brow != out_row).elements());
    ASSERT_EQ(0, where(brows != out_rows).elements());
    ASSERT_EQ(0, where(bcol != out_col).elements());
    ASSERT_EQ(0, where(bcols != out_cols).elements());

    array avol = range(dim4(10, 10, 10));
    array bvol = avol(seq(1, 7), span, span);
    array bslice = bvol.slice(5);
    array bslices = bvol.slices(5, 6);

    array out_slice = avol(seq(1,7), span, span).slice(5);
    array out_slices = avol(seq(1,7), span, span).slices(5, 6);

    ASSERT_EQ(0, where(bslice != out_slice).elements());
    ASSERT_EQ(0, where(bslices != out_slices).elements());
}
#endif

TEST(Indexing, SNIPPET_indexing_first)
{
    using namespace af;
    //! [ex_indexing_first]
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
    af_print(b(seq(1,7,2)));
    af_print(b(seq(0,end,2)));
    //! [ex_indexing_first]


    array lin_first = A(0);
    array lin_last = A(end);
    array lin_snd_last = A(end-1);

    EXPECT_EQ(1, lin_first.dims(0));
    EXPECT_EQ(1, lin_first.elements());
    EXPECT_EQ(1, lin_last.dims(0));
    EXPECT_EQ(1, lin_last.elements());
    EXPECT_EQ(1, lin_snd_last.dims(0));
    EXPECT_EQ(1, lin_snd_last.elements());

    EXPECT_FLOAT_EQ(1.0f, lin_first.scalar<float>());
    EXPECT_FLOAT_EQ(9.0f, lin_last.scalar<float>());
    EXPECT_FLOAT_EQ(8.0f, lin_snd_last.scalar<float>());


    lin_last = A(-1);
    EXPECT_EQ(1, lin_last.dims(0));
    EXPECT_EQ(1, lin_last.elements());
    EXPECT_FLOAT_EQ(9.0f, lin_last.scalar<float>());

    {
        array out = b(seq(3));
        ASSERT_EQ(3, out.elements());
        vector<float> hout(out.elements());
        out.host(&hout.front());
        for(unsigned i = 0; i < hout.size(); i++) { ASSERT_FLOAT_EQ(b_host[i], hout[i]); }
    }

    {
        array out = b(seq(1, 7));
        ASSERT_EQ(7, out.elements());
        vector<float> hout(out.elements());
        out.host(&hout.front());
        for(unsigned i = 1; i < hout.size(); i++) { ASSERT_FLOAT_EQ(b_host[i], hout[i - 1]); }
    }

    {
        array out = b(seq(1, 7, 2));
        ASSERT_EQ(4, out.elements());
        vector<float> hout(out.elements());
        out.host(&hout.front());
        for(unsigned i = 0; i < hout.size(); i++) { ASSERT_FLOAT_EQ(b_host[i * 2 + 1], hout[i]); }
    }
}

TEST(Indexing, SNIPPET_indexing_set)
{
    using namespace af;
    //! [ex_indexing_set]
    array A = constant(0, 3, 3);
    af_print(A);

    // setting entries to a constant
    A(span) = 4;        // fill entire array
    af_print(A);

    A.row(0) = -1;      // first row
    af_print(A);

    A(seq(3)) = 3.1415; // first three elements
    af_print(A);

    // copy in another matrix
    array B = constant(1, 4, 4, s32);
    B.row(0) = randu(1, 4, f32); // set a row to random values (also upcast)
    //! [ex_indexing_set]
    //TODO: Confirm the outputs are correct. see #697
}


TEST(Indexing, SNIPPET_indexing_ref)
{
    using namespace af;
    //! [ex_indexing_ref]
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
    //! [ex_indexing_ref]
    //TODO: Confirm the outputs are correct. see #697
}

TEST(Indexing, SNIPPET_indexing_copy)
{
  af::array A = af::constant(0,1, s32);
  af::index s1;
  s1 = af::index(A);
  // At exit both A and s1 will be destroyed
  // but the underlying array should only be
  // freed once.
}
