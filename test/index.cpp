/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::cout;
using std::endl;
using std::ostream_iterator;
using std::string;
using std::vector;

template<typename T, typename OP>
void checkValues(const af_seq &seq, const T *data, const T *indexed_data,
                 OP compair_op) {
    for (int i = 0, j = seq.begin; compair_op(j, (int)seq.end);
         j += seq.step, i++) {
        ASSERT_DOUBLE_EQ(real(data[j]), real(indexed_data[i]))
            << "Where i = " << i << " and j = " << j;
    }
}

template<typename T>
void DimCheck(const vector<af_seq> &seqs) {
    SUPPORTED_TYPE_CHECK(T);

    static const int ndims   = 1;
    static const size_t dims = 100;

    dim_t d[1] = {dims};

    vector<T> hData(dims);
    for (int i = 0; i < (int)dims; i++) { hData[i] = i; }

    af_array a = 0;
    ASSERT_SUCCESS(af_create_array(&a, &hData.front(), ndims, d,
                                   (af_dtype)dtype_traits<T>::af_type));

    vector<af_array> indexed_array(seqs.size(), 0);
    for (size_t i = 0; i < seqs.size(); i++) {
        ASSERT_SUCCESS(af_index(&(indexed_array[i]), a, ndims, &seqs[i]))
            << "where seqs[i].begin == " << seqs[i].begin
            << " seqs[i].step == " << seqs[i].step
            << " seqs[i].end == " << seqs[i].end;
    }

    vector<T *> h_indexed(seqs.size());
    for (size_t i = 0; i < seqs.size(); i++) {
        dim_t elems;
        ASSERT_SUCCESS(af_get_elements(&elems, indexed_array[i]));
        h_indexed[i] = new T[elems];
        ASSERT_SUCCESS(
            af_get_data_ptr((void *)(h_indexed[i]), indexed_array[i]));
    }

    for (size_t k = 0; k < seqs.size(); k++) {
        if (seqs[k].step > 0) {
            checkValues(seqs[k], &hData.front(), h_indexed[k],
                        std::less_equal<int>());
        } else if (seqs[k].step < 0) {
            checkValues(seqs[k], &hData.front(), h_indexed[k],
                        std::greater_equal<int>());
        } else {
            for (size_t i = 0; i <= seqs[k].end; i++) {
                ASSERT_DOUBLE_EQ(real(hData[i]), real(h_indexed[k][i]))
                    << "Where i = " << i;
            }
        }
        delete[] h_indexed[k];
    }

    ASSERT_SUCCESS(af_release_array(a));
    for (size_t i = 0; i < indexed_array.size(); i++) {
        ASSERT_SUCCESS(af_release_array(indexed_array[i]));
    }
}

template<typename T>
class Indexing1D : public ::testing::Test {
   public:
    virtual void SetUp() {
        continuous_seqs.push_back(af_make_seq(0, 20, 1));   // Begin Continious
        continuous_seqs.push_back(af_make_seq(80, 99, 1));  // End Continious
        continuous_seqs.push_back(af_make_seq(10, 89, 1));  // Mid Continious

        continuous_reverse_seqs.push_back(
            af_make_seq(20, 0, -1));  // Begin Reverse Continious
        continuous_reverse_seqs.push_back(
            af_make_seq(99, 80, -1));  // End Reverse Continious
        continuous_reverse_seqs.push_back(
            af_make_seq(89, 10, -1));  // Mid Reverse Continious

        strided_seqs.push_back(af_make_seq(5, 40, 2));  // Two Step
        strided_seqs.push_back(af_make_seq(5, 40, 3));  // Three Step
        strided_seqs.push_back(af_make_seq(5, 40, 4));  // Four Step

        strided_reverse_seqs.push_back(
            af_make_seq(40, 5, -2));  // Reverse Two Step
        strided_reverse_seqs.push_back(
            af_make_seq(40, 5, -3));  // Reverse Three Step
        strided_reverse_seqs.push_back(
            af_make_seq(40, 5, -4));  // Reverse Four Step

        span_seqs.push_back(af_span);
    }

    virtual ~Indexing1D() {}

    // virtual void TearDown() {}

    vector<af_seq> continuous_seqs;
    vector<af_seq> continuous_reverse_seqs;
    vector<af_seq> strided_seqs;
    vector<af_seq> strided_reverse_seqs;
    vector<af_seq> span_seqs;
};

typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned,
                         unsigned char, intl, uintl, short, ushort,
                         half_float::half>
    AllTypes;
TYPED_TEST_SUITE(Indexing1D, AllTypes);

TYPED_TEST(Indexing1D, Continious) {
    DimCheck<TypeParam>(this->continuous_seqs);
}
TYPED_TEST(Indexing1D, ContiniousReverse) {
    DimCheck<TypeParam>(this->continuous_reverse_seqs);
}
TYPED_TEST(Indexing1D, Strided) { DimCheck<TypeParam>(this->strided_seqs); }
TYPED_TEST(Indexing1D, StridedReverse) {
    DimCheck<TypeParam>(this->strided_reverse_seqs);
}
TYPED_TEST(Indexing1D, Span) { DimCheck<TypeParam>(this->span_seqs); }

template<typename T>
class Indexing2D : public ::testing::Test {
   public:
    vector<af_seq> make_vec(af_seq first, af_seq second) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        return out;
    }
    virtual void SetUp() {
        column_continuous_seq.push_back(
            make_vec(af_span, af_make_seq(0, 6, 1)));
        column_continuous_seq.push_back(
            make_vec(af_span, af_make_seq(4, 9, 1)));
        column_continuous_seq.push_back(
            make_vec(af_span, af_make_seq(3, 8, 1)));

        column_continuous_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(6, 0, -1)));
        column_continuous_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(9, 4, -1)));
        column_continuous_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(8, 3, -1)));

        column_strided_seq.push_back(
            make_vec(af_span, af_make_seq(0, 8, 2)));  // Two Step
        column_strided_seq.push_back(
            make_vec(af_span, af_make_seq(2, 9, 3)));  // Three Step
        column_strided_seq.push_back(
            make_vec(af_span, af_make_seq(0, 9, 4)));  // Four Step

        column_strided_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(8, 0, -2)));  // Two Step
        column_strided_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(9, 2, -3)));  // Three Step
        column_strided_reverse_seq.push_back(
            make_vec(af_span, af_make_seq(9, 0, -4)));  // Four Step

        row_continuous_seq.push_back(make_vec(af_make_seq(0, 6, 1), af_span));
        row_continuous_seq.push_back(make_vec(af_make_seq(4, 9, 1), af_span));
        row_continuous_seq.push_back(make_vec(af_make_seq(3, 8, 1), af_span));

        row_continuous_reverse_seq.push_back(
            make_vec(af_make_seq(6, 0, -1), af_span));
        row_continuous_reverse_seq.push_back(
            make_vec(af_make_seq(9, 4, -1), af_span));
        row_continuous_reverse_seq.push_back(
            make_vec(af_make_seq(8, 3, -1), af_span));

        row_strided_seq.push_back(make_vec(af_make_seq(0, 8, 2), af_span));
        row_strided_seq.push_back(make_vec(af_make_seq(2, 9, 3), af_span));
        row_strided_seq.push_back(make_vec(af_make_seq(0, 9, 4), af_span));

        row_strided_reverse_seq.push_back(
            make_vec(af_make_seq(8, 0, -2), af_span));
        row_strided_reverse_seq.push_back(
            make_vec(af_make_seq(9, 2, -3), af_span));
        row_strided_reverse_seq.push_back(
            make_vec(af_make_seq(9, 0, -4), af_span));

        continuous_continuous_seq.push_back(
            make_vec(af_make_seq(1, 6, 1), af_make_seq(0, 6, 1)));
        continuous_continuous_seq.push_back(
            make_vec(af_make_seq(3, 9, 1), af_make_seq(4, 9, 1)));
        continuous_continuous_seq.push_back(
            make_vec(af_make_seq(5, 8, 1), af_make_seq(3, 8, 1)));

        continuous_reverse_seq.push_back(
            make_vec(af_make_seq(1, 6, 1), af_make_seq(6, 0, -1)));
        continuous_reverse_seq.push_back(
            make_vec(af_make_seq(3, 9, 1), af_make_seq(9, 4, -1)));
        continuous_reverse_seq.push_back(
            make_vec(af_make_seq(5, 8, 1), af_make_seq(8, 3, -1)));

        continuous_strided_seq.push_back(
            make_vec(af_make_seq(1, 6, 1), af_make_seq(0, 8, 2)));
        continuous_strided_seq.push_back(
            make_vec(af_make_seq(3, 9, 1), af_make_seq(2, 9, 3)));
        continuous_strided_seq.push_back(
            make_vec(af_make_seq(5, 8, 1), af_make_seq(1, 9, 4)));

        continuous_strided_reverse_seq.push_back(
            make_vec(af_make_seq(1, 6, 1), af_make_seq(8, 0, -2)));
        continuous_strided_reverse_seq.push_back(
            make_vec(af_make_seq(3, 9, 1), af_make_seq(9, 2, -3)));
        continuous_strided_reverse_seq.push_back(
            make_vec(af_make_seq(5, 8, 1), af_make_seq(9, 1, -4)));

        reverse_continuous_seq.push_back(
            make_vec(af_make_seq(6, 1, -1), af_make_seq(0, 6, 1)));
        reverse_continuous_seq.push_back(
            make_vec(af_make_seq(9, 3, -1), af_make_seq(4, 9, 1)));
        reverse_continuous_seq.push_back(
            make_vec(af_make_seq(8, 5, -1), af_make_seq(3, 8, 1)));

        reverse_reverse_seq.push_back(
            make_vec(af_make_seq(6, 1, -1), af_make_seq(6, 0, -1)));
        reverse_reverse_seq.push_back(
            make_vec(af_make_seq(9, 3, -1), af_make_seq(9, 4, -1)));
        reverse_reverse_seq.push_back(
            make_vec(af_make_seq(8, 5, -1), af_make_seq(8, 3, -1)));

        reverse_strided_seq.push_back(
            make_vec(af_make_seq(6, 1, -1), af_make_seq(0, 8, 2)));
        reverse_strided_seq.push_back(
            make_vec(af_make_seq(9, 3, -1), af_make_seq(2, 9, 3)));
        reverse_strided_seq.push_back(
            make_vec(af_make_seq(8, 5, -1), af_make_seq(1, 9, 4)));

        reverse_strided_reverse_seq.push_back(
            make_vec(af_make_seq(6, 1, -1), af_make_seq(8, 0, -2)));
        reverse_strided_reverse_seq.push_back(
            make_vec(af_make_seq(9, 3, -1), af_make_seq(9, 2, -3)));
        reverse_strided_reverse_seq.push_back(
            make_vec(af_make_seq(8, 5, -1), af_make_seq(9, 1, -4)));

        strided_continuous_seq.push_back(
            make_vec(af_make_seq(0, 8, 2), af_make_seq(0, 6, 1)));
        strided_continuous_seq.push_back(
            make_vec(af_make_seq(2, 9, 3), af_make_seq(4, 9, 1)));
        strided_continuous_seq.push_back(
            make_vec(af_make_seq(1, 9, 4), af_make_seq(3, 8, 1)));

        strided_strided_seq.push_back(
            make_vec(af_make_seq(1, 6, 2), af_make_seq(0, 8, 2)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(3, 9, 2), af_make_seq(2, 9, 3)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(5, 8, 2), af_make_seq(1, 9, 4)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(1, 6, 3), af_make_seq(0, 8, 2)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(3, 9, 3), af_make_seq(2, 9, 3)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(5, 8, 3), af_make_seq(1, 9, 4)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(1, 6, 4), af_make_seq(0, 8, 2)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(3, 9, 4), af_make_seq(2, 9, 3)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(3, 8, 4), af_make_seq(1, 9, 4)));
        strided_strided_seq.push_back(
            make_vec(af_make_seq(3, 6, 4), af_make_seq(1, 9, 4)));
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

template<typename T>
void DimCheck2D(const vector<vector<af_seq>> &seqs, string TestFile,
                size_t NDims) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> hData;
    vector<vector<T>> tests;
    readTests<T, T, int>(TestFile, numDims, hData, tests);
    dim4 dimensions = numDims[0];

    af_array a = 0;
    ASSERT_SUCCESS(af_create_array(&a, &(hData[0].front()), NDims,
                                   dimensions.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    vector<af_array> indexed_arrays(seqs.size(), 0);
    for (size_t i = 0; i < seqs.size(); i++) {
        ASSERT_SUCCESS(
            af_index(&(indexed_arrays[i]), a, NDims, seqs[i].data()));
    }

    vector<T *> h_indexed(seqs.size(), NULL);
    for (size_t i = 0; i < seqs.size(); i++) {
        dim_t elems;
        ASSERT_SUCCESS(af_get_elements(&elems, indexed_arrays[i]));
        h_indexed[i] = new T[elems];
        ASSERT_SUCCESS(
            af_get_data_ptr((void *)h_indexed[i], indexed_arrays[i]));

        T *ptr = h_indexed[i];
        if (false == equal(ptr, ptr + tests[i].size(), tests[i].begin())) {
            cout << "index data: ";
            copy(ptr, ptr + tests[i].size(), ostream_iterator<T>(cout, ", "));
            cout << endl << "file data: ";
            copy(tests[i].begin(), tests[i].end(),
                 ostream_iterator<T>(cout, ", "));
            FAIL() << "indexed_array[" << i << "] FAILED" << endl;
        }
        delete[] h_indexed[i];
    }

    ASSERT_SUCCESS(af_release_array(a));
    for (size_t i = 0; i < indexed_arrays.size(); i++) {
        ASSERT_SUCCESS(af_release_array(indexed_arrays[i]));
    }
}

TYPED_TEST_SUITE(Indexing2D, AllTypes);

TYPED_TEST(Indexing2D, ColumnContinious) {
    DimCheck2D<TypeParam>(this->column_continuous_seq,
                          TEST_DIR "/index/ColumnContinious.test", 2);
}

TYPED_TEST(Indexing2D, ColumnContiniousReverse) {
    DimCheck2D<TypeParam>(this->column_continuous_reverse_seq,
                          TEST_DIR "/index/ColumnContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, ColumnStrided) {
    DimCheck2D<TypeParam>(this->column_strided_seq,
                          TEST_DIR "/index/ColumnStrided.test", 2);
}

TYPED_TEST(Indexing2D, ColumnStridedReverse) {
    DimCheck2D<TypeParam>(this->column_strided_reverse_seq,
                          TEST_DIR "/index/ColumnStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, RowContinious) {
    DimCheck2D<TypeParam>(this->row_continuous_seq,
                          TEST_DIR "/index/RowContinious.test", 2);
}

TYPED_TEST(Indexing2D, RowContiniousReverse) {
    DimCheck2D<TypeParam>(this->row_continuous_reverse_seq,
                          TEST_DIR "/index/RowContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, RowStrided) {
    DimCheck2D<TypeParam>(this->row_strided_seq,
                          TEST_DIR "/index/RowStrided.test", 2);
}

TYPED_TEST(Indexing2D, RowStridedReverse) {
    DimCheck2D<TypeParam>(this->row_strided_reverse_seq,
                          TEST_DIR "/index/RowStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousContinious) {
    DimCheck2D<TypeParam>(this->continuous_continuous_seq,
                          TEST_DIR "/index/ContiniousContinious.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousReverse) {
    DimCheck2D<TypeParam>(this->continuous_reverse_seq,
                          TEST_DIR "/index/ContiniousReverse.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousStrided) {
    DimCheck2D<TypeParam>(this->continuous_strided_seq,
                          TEST_DIR "/index/ContiniousStrided.test", 2);
}

TYPED_TEST(Indexing2D, ContiniousStridedReverse) {
    DimCheck2D<TypeParam>(this->continuous_strided_reverse_seq,
                          TEST_DIR "/index/ContiniousStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, ReverseContinious) {
    DimCheck2D<TypeParam>(this->reverse_continuous_seq,
                          TEST_DIR "/index/ReverseContinious.test", 2);
}

TYPED_TEST(Indexing2D, ReverseReverse) {
    DimCheck2D<TypeParam>(this->reverse_reverse_seq,
                          TEST_DIR "/index/ReverseReverse.test", 2);
}

TYPED_TEST(Indexing2D, ReverseStrided) {
    DimCheck2D<TypeParam>(this->reverse_strided_seq,
                          TEST_DIR "/index/ReverseStrided.test", 2);
}

TYPED_TEST(Indexing2D, ReverseStridedReverse) {
    DimCheck2D<TypeParam>(this->reverse_strided_reverse_seq,
                          TEST_DIR "/index/ReverseStridedReverse.test", 2);
}

TYPED_TEST(Indexing2D, StridedContinious) {
    DimCheck2D<TypeParam>(this->strided_continuous_seq,
                          TEST_DIR "/index/StridedContinious.test", 2);
}

TYPED_TEST(Indexing2D, StridedStrided) {
    DimCheck2D<TypeParam>(this->strided_strided_seq,
                          TEST_DIR "/index/StridedStrided.test", 2);
}

vector<af_seq> make_vec(af_seq first, af_seq second) {
    vector<af_seq> out;
    out.push_back(first);
    out.push_back(second);
    return out;
}

template<typename T>
class Indexing : public ::testing::Test {
    vector<af_seq> make_vec3(af_seq first, af_seq second, af_seq third) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        out.push_back(third);
        return out;
    }

    vector<af_seq> make_vec4(af_seq first, af_seq second, af_seq third,
                             af_seq fourth) {
        vector<af_seq> out;
        out.push_back(first);
        out.push_back(second);
        out.push_back(third);
        out.push_back(fourth);
        return out;
    }

   public:
    virtual void SetUp() {
        continuous3d_to_3d.push_back(
            make_vec3(af_make_seq(0, 4, 1), af_make_seq(0, 6, 1), af_span));
        continuous3d_to_3d.push_back(
            make_vec3(af_make_seq(4, 8, 1), af_make_seq(4, 9, 1), af_span));
        continuous3d_to_3d.push_back(
            make_vec3(af_make_seq(6, 9, 1), af_make_seq(3, 8, 1), af_span));

        continuous3d_to_2d.push_back(
            make_vec3(af_span, af_make_seq(0, 6, 1), af_make_seq(0, 0, 1)));
        continuous3d_to_2d.push_back(
            make_vec3(af_span, af_make_seq(4, 9, 1), af_make_seq(1, 1, 1)));
        continuous3d_to_2d.push_back(
            make_vec3(af_span, af_make_seq(3, 8, 1), af_make_seq(0, 0, 1)));

        continuous3d_to_1d.push_back(
            make_vec3(af_span, af_make_seq(0, 0, 1), af_make_seq(0, 0, 1)));
        continuous3d_to_1d.push_back(
            make_vec3(af_span, af_make_seq(6, 6, 1), af_make_seq(1, 1, 1)));
        continuous3d_to_1d.push_back(
            make_vec3(af_span, af_make_seq(9, 9, 1), af_make_seq(0, 0, 1)));

        continuous4d_to_4d.push_back(make_vec4(
            af_make_seq(2, 6, 1), af_make_seq(2, 6, 1), af_span, af_span));
        continuous4d_to_3d.push_back(make_vec4(af_make_seq(2, 6, 1),
                                               af_make_seq(2, 6, 1), af_span,
                                               af_make_seq(0, 0, 1)));
        continuous4d_to_2d.push_back(
            make_vec4(af_make_seq(2, 6, 1), af_make_seq(2, 6, 1),
                      af_make_seq(0, 0, 1), af_make_seq(0, 0, 1)));
        continuous4d_to_1d.push_back(
            make_vec4(af_make_seq(2, 6, 1), af_make_seq(2, 2, 1),
                      af_make_seq(0, 0, 1), af_make_seq(0, 0, 1)));
    }

    vector<vector<af_seq>> continuous3d_to_3d;
    vector<vector<af_seq>> continuous3d_to_2d;
    vector<vector<af_seq>> continuous3d_to_1d;

    vector<vector<af_seq>> continuous4d_to_4d;
    vector<vector<af_seq>> continuous4d_to_3d;
    vector<vector<af_seq>> continuous4d_to_2d;
    vector<vector<af_seq>> continuous4d_to_1d;
};

template<typename T>
void DimCheckND(const vector<vector<af_seq>> &seqs, string TestFile,
                size_t NDims) {
    SUPPORTED_TYPE_CHECK(T);

    // DimCheck2D function is generalized enough
    // to check 3d and 4d indexing
    DimCheck2D<T>(seqs, TestFile, NDims);
}

TYPED_TEST_SUITE(Indexing, AllTypes);

TYPED_TEST(Indexing, 4D_to_4D) {
    DimCheckND<TypeParam>(this->continuous4d_to_4d,
                          TEST_DIR "/index/Continuous4Dto4D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_3D) {
    DimCheckND<TypeParam>(this->continuous4d_to_3d,
                          TEST_DIR "/index/Continuous4Dto3D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_2D) {
    DimCheckND<TypeParam>(this->continuous4d_to_2d,
                          TEST_DIR "/index/Continuous4Dto2D.test", 4);
}

TYPED_TEST(Indexing, 4D_to_1D) {
    DimCheckND<TypeParam>(this->continuous4d_to_1d,
                          TEST_DIR "/index/Continuous4Dto1D.test", 4);
}

TYPED_TEST(Indexing, 3D_to_3D) {
    DimCheckND<TypeParam>(this->continuous3d_to_3d,
                          TEST_DIR "/index/Continuous3Dto3D.test", 3);
}

TYPED_TEST(Indexing, 3D_to_2D) {
    DimCheckND<TypeParam>(this->continuous3d_to_2d,
                          TEST_DIR "/index/Continuous3Dto2D.test", 3);
}

TYPED_TEST(Indexing, 3D_to_1D) {
    DimCheckND<TypeParam>(this->continuous3d_to_1d,
                          TEST_DIR "/index/Continuous3Dto1D.test", 3);
}

TEST(Index, Docs_Util_C_API) {
    //![ex_index_util_0]
    af_index_t *indexers = 0;
    af_err err           = af_create_indexers(
                  &indexers);  // Memory is allocated on heap by the callee
    // by default all the indexers span all the elements along the given
    // dimension

    // Create array
    af_array a;
    unsigned ndims = 2;
    dim_t dim[]    = {10, 10};
    af_randu(&a, ndims, dim, f32);

    // Create index array
    af_array idx;
    unsigned n = 1;
    dim_t d[]  = {5};
    af_range(&idx, n, d, 0, s32);

    af_print_array(a);
    af_print_array(idx);

    // create array indexer
    err = af_set_array_indexer(indexers, idx, 1);

    // index with indexers
    af_array out;
    af_index_gen(&out, a, 2,
                 indexers);  // number of indexers should be two since
                             // we have set only second af_index_t
    if (err != AF_SUCCESS) {
        printf("Failed in af_index_gen: %d\n", err);
        throw;
    }
    af_print_array(out);
    af_release_array(out);

    af_seq zeroIndices = af_make_seq(0.0, 9.0, 2.0);

    err = af_set_seq_indexer(indexers, &zeroIndices, 0, false);

    err = af_index_gen(&out, a, 2, indexers);
    if (err != AF_SUCCESS) {
        printf("Failed in af_index_gen: %d\n", err);
        throw;
    }
    af_print_array(out);

    af_release_indexers(indexers);
    af_release_array(a);
    af_release_array(idx);
    af_release_array(out);
    //![ex_index_util_0]
}

//////////////////////////////// CPP ////////////////////////////////

using af::allTrue;
using af::array;
using af::constant;
using af::deviceGC;
using af::deviceMemInfo;
using af::end;
using af::freeHost;
using af::randu;
using af::range;
using af::reorder;
using af::seq;
using af::span;
using af::where;

TEST(Indexing2D, ColumnContiniousCPP) {
    vector<vector<af_seq>> seqs;

    seqs.push_back(make_vec(af_span, af_make_seq(0, 6, 1)));
    // seqs.push_back(make_vec(span, af_make_seq(  4,  9,  1)));
    // seqs.push_back(make_vec(span, af_make_seq(  3,  8,  1)));

    vector<dim4> numDims;

    vector<vector<float>> hData;
    vector<vector<float>> tests;
    readTests<float, float, int>(TEST_DIR "/index/ColumnContinious.test",
                                 numDims, hData, tests);
    dim4 dimensions = numDims[0];

    array a(dimensions, &(hData[0].front()));

    vector<array> sub;
    for (size_t i = 0; i < seqs.size(); i++) {
        vector<af_seq> seq = seqs[i];
        sub.push_back(a(seq[0], seq[1]));
    }

    for (size_t i = 0; i < seqs.size(); i++) {
        dim_t elems = sub[i].elements();
        float *ptr  = new float[elems];
        sub[i].host(ptr);

        if (false == equal(ptr, ptr + tests[i].size(), tests[i].begin())) {
            cout << "index data: ";
            copy(ptr, ptr + tests[i].size(),
                 ostream_iterator<float>(cout, ", "));
            cout << endl << "file data: ";
            copy(tests[i].begin(), tests[i].end(),
                 ostream_iterator<float>(cout, ", "));
            FAIL() << "indexed_array[" << i << "] FAILED" << endl;
        }
        delete[] ptr;
    }
}

/************************ Array Based indexing tests from here on
 * ******************/

template<typename T>
class lookup : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double, int, unsigned, unsigned char, short,
                         ushort, intl, uintl, half_float::half>
    ArrIdxTestTypes;
TYPED_TEST_SUITE(lookup, ArrIdxTestTypes);

template<typename T>
void arrayIndexTest(string pTestFile, int dim) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 dims0        = numDims[0];
    dim4 dims1        = numDims[1];
    af_array outArray = 0;
    af_array inArray  = 0;
    af_array idxArray = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&idxArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_lookup(&outArray, inArray, idxArray, dim));

    vector<T> currGoldBar = tests[0];
    dim4 goldDims         = dims0;
    goldDims[dim]         = dims1[0];

    ASSERT_VEC_ARRAY_EQ(currGoldBar, goldDims, outArray);

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(idxArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(lookup, Dim0) {
    arrayIndexTest<TypeParam>(string(TEST_DIR "/arrayindex/dim0.test"), 0);
}

TYPED_TEST(lookup, Dim1) {
    arrayIndexTest<TypeParam>(string(TEST_DIR "/arrayindex/dim1.test"), 1);
}

TYPED_TEST(lookup, Dim2) {
    arrayIndexTest<TypeParam>(string(TEST_DIR "/arrayindex/dim2.test"), 2);
}

TYPED_TEST(lookup, Dim3) {
    arrayIndexTest<TypeParam>(string(TEST_DIR "/arrayindex/dim3.test"), 3);
}

TEST(lookup, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(string(TEST_DIR "/arrayindex/dim0.test"),
                                 numDims, in, tests);

    dim4 dims0 = numDims[0];
    dim4 dims1 = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = af::lookup(input, indices, 0);

    vector<float> currGoldBar = tests[0];
    dim4 goldDims             = dims0;
    goldDims[0]               = dims1[0];

    ASSERT_VEC_ARRAY_EQ(currGoldBar, goldDims, output);
}

TEST(lookup, largeDim) {
    const size_t largeDim = 65535 * 8 + 1;

    cleanSlate();
    array input   = range(dim4(2, largeDim));
    array indices = constant(1, 100);

    array output = af::lookup(input, indices);
}

TEST(lookup, Issue2009) {
    array a   = range(dim4(1000, 1));
    array idx = constant(0, 1, u32);
    array b   = af::lookup(a, idx, 1);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST(lookup, SNIPPET_lookup1d) {
    //! [ex_index_lookup1d]

    // input array
    float in_[5] = {10, 20, 30, 40, 50};
    af::array in(5, in_);

    // indices to lookup
    int idx_[3] = {1, 3, 2};
    af::array idx(3, idx_);

    af::array indexed = af::lookup(in, idx);
    // indexed == { 20, 40, 30 };

    //! [ex_index_lookup1d]

    // indexing tests
    float in_g[3] = {20, 40, 30};
    af::array indexed_gold(3, in_g);
    ASSERT_ARRAYS_NEAR(indexed, indexed_gold, 1e-5);
}

TEST(lookup, SNIPPET_lookup_oob) {
    //! [ex_index_lookup_oob]

    // input array
    float in_[5] = {10, 20, 30, 40, 50};
    af::array in(5, in_);

    // indexing past end of array
    int idx_outofbounds_p_[8] = {4, 5, 6, 7, 8, 9, 10, 11};
    af::array idx_outofbounds_p(8, idx_outofbounds_p_);

    // and indexing before beginning of array
    int idx_outofbounds_n_[8] = {0, -1, -2, -3, -4, -5, -6, -7};
    af::array idx_outofbounds_n(8, idx_outofbounds_n_);

    af::array indexed_out_of_bounds_pos = af::lookup(in, idx_outofbounds_p);
    af::array indexed_out_of_bounds_neg = af::lookup(in, idx_outofbounds_n);
    // indexed_out_of_bounds_pos == { 50, 50, 40, 30, 20, 10, 50, 40 }
    // indexed_out_of_bounds_neg == { 10, 10, 20, 30, 40, 50, 10, 20 }

    //! [ex_index_lookup_oob]

    // out of bounds tests
    float oob_p_g_[8] = {50, 50, 40, 30, 20, 10, 50, 40};
    af::array oob_p_g(8, oob_p_g_);
    ASSERT_ARRAYS_NEAR(indexed_out_of_bounds_pos, oob_p_g, 1e-5);
    float oob_n_g_[8] = {10, 10, 20, 30, 40, 50, 10, 20};
    af::array oob_n_g(8, oob_n_g_);
    ASSERT_ARRAYS_NEAR(indexed_out_of_bounds_neg, oob_n_g, 1e-5);
}

TEST(lookup, SNIPPET_lookup2d) {
    //! [ex_index_lookup2d]

    // constant input data
    float input_vals[9] = {10, 20, 30, 11, 21, 31, 12, 22, 32};
    array input(3, 3, input_vals);
    // {{10 11 12},
    //  {20 21 22},
    //  {30 31 32}},

    // indices to lookup
    int idx_[6] = {0, 0, 1, 1, 2, 2};
    af::array idx(6, idx_);

    // will look up all indices along specified dimension
    af::array indexed = af::lookup(input, idx);  //(dim = 0)
    // indexed == { 10, 11, 12,
    //              10, 11, 12,
    //              20, 21, 22,
    //              20, 21, 22,
    //              30, 31, 32,
    //              30, 31, 32 };

    af::array indexed_dim1 = af::lookup(input, idx, 1);
    // indexed_dim1 == { 10, 10, 11, 11, 12, 12,
    //                   20, 20, 21, 21, 22, 22,
    //                   30, 30, 31, 31, 32, 32 };

    //! [ex_index_lookup2d]

    float expected_indexed[18] = {10, 10, 20, 20, 30, 30, 11, 11, 21,
                                  21, 31, 31, 12, 12, 22, 22, 32, 32};

    array indexed_gold(6, 3, expected_indexed);
    ASSERT_ARRAYS_NEAR(indexed, indexed_gold, 1e-5);

    float expected_indexed_dim1[18] = {10, 20, 30, 10, 20, 30, 11, 21, 31,
                                       11, 21, 31, 12, 22, 32, 12, 22, 32};

    array indexed_gold_dim1(3, 6, expected_indexed_dim1);
    ASSERT_ARRAYS_NEAR(indexed_dim1, indexed_gold_dim1, 1e-5);
}

TEST(SeqIndex, CPP_END) {
    const int n       = 5;
    const int m       = 5;
    const int end_off = 2;

    array a = randu(n, m);
    array b = a(end - end_off, span);

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < m; i++) { ASSERT_EQ(hA[i * n + end_off], hB[i]); }

    freeHost(hA);
    freeHost(hB);
}

TEST(SeqIndex, CPP_END_SEQ) {
    const int num       = 20;
    const int end_begin = 10;
    const int end_end   = 0;

    array a = randu(num);
    array b = a(seq(end - end_begin, end - end_end));

    float *hA = a.host<float>();
    float *hB = b.host<float>();

    for (int i = 0; i < end_begin - end_end + 1; i++) {
        ASSERT_EQ(hA[i + end_begin - 1], hB[i]);
    }

    freeHost(hA);
    freeHost(hB);
}

array cpp_scope_seq_test(const int num, const float val, const seq s) {
    array a = constant(val, num);
    return a(s);
}

TEST(SeqIndex, CPP_SCOPE_SEQ) {
    const int num       = 20;
    const int seq_begin = 3;
    const int seq_end   = 10;
    const float val     = 133.33;

    array b   = cpp_scope_seq_test(num, val, seq(seq_begin, seq_end));
    float *hB = b.host<float>();

    for (int i = 0; i < seq_end - seq_begin + 1; i++) { ASSERT_EQ(hB[i], val); }

    freeHost(hB);
}

array cpp_scope_arr_test(const int num, const float val) {
    array a   = constant(val, num);
    array idx = where(a > val / 2);
    return a(idx) * (val - 1);
}

TEST(SeqIndex, CPP_SCOPE_ARR) {
    const int num   = 20;
    const float val = 133.33;

    array b   = cpp_scope_arr_test(num, val);
    float *hB = b.host<float>();

    for (int i = 0; i < (int)b.elements(); i++) {
        ASSERT_EQ(hB[i], val * (val - 1));
    }

    freeHost(hB);
}

TEST(SeqIndex, CPPLarge) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(string(TEST_DIR "/arrayindex/dim0Large.test"),
                                 numDims, in, tests);

    dim4 dims0 = numDims[0];
    dim4 dims1 = numDims[1];

    array input(dims0, &(in[0].front()));
    array indices(dims1, &(in[1].front()));
    array output = af::lookup(input, indices, 0);

    vector<float> currGoldBar = tests[0];
    dim4 goldDims             = dims0;
    goldDims[0]               = dims1[0];

    ASSERT_VEC_ARRAY_EQ(currGoldBar, goldDims, output);
}

TEST(SeqIndex, Cascade00) {
    const int nx = 200;
    const int ny = 200;

    const int stb = 21;
    const int enb = 180;

    const int stc = 3;    // Should be less than nx - stb
    const int enc = 109;  // Should be less than ny - enb

    const int st  = stb + stc;
    const int en  = stb + enc;
    const int nxc = en - st + 1;

    array a = randu(nx, ny);
    array b = a(seq(stb, enb), span);
    array c = b(seq(stc, enc), span);

    ASSERT_EQ(c.dims(1), (dim_t)ny);
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = 0; j < ny; j++) {
        int a_off = j * nx;
        int c_off = j * nxc;

        for (int i = st; i < en; i++) {
            ASSERT_EQ(h_a[a_off + i], h_c[c_off + i - st])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_a);
    freeHost(h_b);
    freeHost(h_c);
}

TEST(SeqIndex, Cascade01) {
    const int nx = 200;
    const int ny = 200;

    const int stb = 54;
    const int enb = 196;

    const int stc = 39;
    const int enc = 123;

    const int nxc = enb - stb + 1;
    const int nyc = enc - stc + 1;

    array a = randu(nx, ny);
    array b = a(seq(stb, enb), span);
    array c = b(span, seq(stc, enc));

    ASSERT_EQ(c.dims(1), (dim_t)nyc);
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = stc; j < enc; j++) {
        int a_off = j * nx;
        int c_off = (j - stc) * nxc;

        for (int i = stb; i < enb; i++) {
            ASSERT_EQ(h_a[a_off + i], h_c[c_off + i - stb])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_a);
    freeHost(h_b);
    freeHost(h_c);
}

TEST(SeqIndex, Cascade10) {
    const int nx = 200;
    const int ny = 200;

    const int stb = 71;
    const int enb = 188;

    const int stc = 33;
    const int enc = 155;

    const int nxc = enc - stc + 1;
    const int nyc = enb - stb + 1;

    array a = randu(nx, ny);
    array b = a(span, seq(stb, enb));
    array c = b(seq(stc, enc), span);

    ASSERT_EQ(c.dims(1), (dim_t)nyc);
    ASSERT_EQ(c.dims(0), (dim_t)nxc);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = stb; j < enb; j++) {
        int a_off = j * nx;
        int c_off = (j - stb) * nxc;

        for (int i = stc; i < enc; i++) {
            ASSERT_EQ(h_a[a_off + i], h_c[c_off + i - stc])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_a);
    freeHost(h_b);
    freeHost(h_c);
}

TEST(SeqIndex, Cascade11) {
    const int nx = 200;
    const int ny = 200;

    const int stb = 50;
    const int enb = 150;

    const int stc = 20;  // Should be less than nx - stb
    const int enc = 80;  // Should be less than ny - enb

    const int st  = stb + stc;
    const int en  = stb + enc;
    const int nyc = en - st + 1;

    array a = randu(nx, ny);
    array b = a(span, seq(stb, enb));
    array c = b(span, seq(stc, enc));

    ASSERT_EQ(c.dims(1), nyc);
    ASSERT_EQ(c.dims(0), nx);

    float *h_a = a.host<float>();
    float *h_b = b.host<float>();
    float *h_c = c.host<float>();

    for (int j = st; j < en; j++) {
        int a_off = j * nx;
        int c_off = (j - st) * nx;

        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_a[a_off + i], h_c[c_off + i])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_a);
    freeHost(h_b);
    freeHost(h_c);
}

TEST(ArrayIndex, CPP_INDEX_VECTOR) {
    float h_inds[] = {0, 3, 2, 1};  // zero-based indexing
    array inds(1, 4, h_inds);
    array B = randu(1, 4);
    array C = B(inds);

    ASSERT_EQ(B.dims(0), 1);
    ASSERT_EQ(B.dims(1), 4);
    ASSERT_EQ(C.dims(0), 1);
    ASSERT_EQ(C.dims(1), 4);

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < 4; i++) { ASSERT_EQ(h_C[i], h_B[(int)h_inds[i]]); }

    freeHost(h_B);
    freeHost(h_C);
}

TEST(SeqIndex, CPP_INDEX_VECTOR) {
    const int num = 20;
    const int len = 10;
    const int st  = 3;
    const int en  = st + len - 1;

    array B = randu(1, 20);
    array C = B(seq(st, en));

    ASSERT_EQ(1, B.dims(0));
    ASSERT_EQ(num, B.dims(1));
    ASSERT_EQ(1, C.dims(0));
    ASSERT_EQ(len, C.dims(1));

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < len; i++) { ASSERT_EQ(h_C[i], h_B[i + st]); }

    freeHost(h_B);
    freeHost(h_C);
}

TEST(ArrayIndex, CPP_INDEX_VECTOR_2D) {
    float h_inds[] = {3, 5, 7, 2};  // zero-based indexing
    array inds(1, 4, h_inds);
    array B = randu(4, 4);
    array C = B(inds);

    ASSERT_EQ(B.dims(0), 4);
    ASSERT_EQ(B.dims(1), 4);
    ASSERT_EQ(C.dims(0), 4);
    ASSERT_EQ(C.dims(1), 1);

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < 4; i++) { ASSERT_EQ(h_C[i], h_B[(int)h_inds[i]]); }

    freeHost(h_B);
    freeHost(h_C);
}

TEST(SeqIndex, CPP_INDEX_VECTOR_2D) {
    const int nx  = 4;
    const int ny  = 3 * nx;
    const int len = 2 * nx;
    const int st  = nx - 1;
    const int en  = st + len - 1;

    array B = randu(nx, ny);
    array C = B(seq(st, en));

    ASSERT_EQ(nx, B.dims(0));
    ASSERT_EQ(ny, B.dims(1));
    ASSERT_EQ(len, C.dims(0));
    ASSERT_EQ(1, C.dims(1));

    float *h_B = B.host<float>();
    float *h_C = C.host<float>();

    for (int i = 0; i < len; i++) { ASSERT_EQ(h_C[i], h_B[i + st]); }

    freeHost(h_B);
    freeHost(h_C);
}

template<typename T>
class IndexedMembers : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

TYPED_TEST_SUITE(IndexedMembers, AllTypes);

TYPED_TEST(IndexedMembers, MemFuncs) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    const dim_t dimsize = 100;
    vector<TypeParam> in(dimsize * dimsize);
    for (int i = 0; i < (int)in.size(); i++) in[i] = i;
    array input(dimsize, dimsize, &in.front(), afHost);

    ASSERT_EQ(dimsize, input(span, 1).elements());
    ASSERT_EQ(input.type(), input(span, 1).type());
    ASSERT_EQ(dim4(dimsize), input(span, 1).dims());
    ASSERT_EQ(1u, input(span, 1).numdims());
    ASSERT_FALSE(input(span, 1).isempty());
    ASSERT_FALSE(input(span, 1).isscalar());
    ASSERT_TRUE(input(1, 1).isscalar());
    ASSERT_TRUE(input(span, 1).isvector());
    ASSERT_FALSE(input(span, 1).isrow());
    ASSERT_EQ(input.iscomplex(), input(span, 1).iscomplex());
    ASSERT_EQ(input.isdouble(), input(span, 1).isdouble());
    ASSERT_EQ(input.issingle(), input(span, 1).issingle());
    ASSERT_EQ(input.isrealfloating(), input(span, 1).isrealfloating());
    ASSERT_EQ(input.isfloating(), input(span, 1).isfloating());
    ASSERT_EQ(input.isinteger(), input(span, 1).isinteger());
    ASSERT_EQ(input.isbool(), input(span, 1).isbool());
    // TODO: Doesn't compile in cuda for cfloat and cdouble
    // ASSERT_EQ(input.scalar<TypeParam>(), input(span, 0).scalar<TypeParam>());
}

#if 1
TYPED_TEST(IndexedMembers, MemIndex) {
    array a     = range(dim4(10, 10));
    array b     = a(seq(1, 7), span);
    array brow  = b.row(5);
    array brows = b.rows(5, 6);
    array bcol  = b.col(5);
    array bcols = b.cols(5, 6);

    array out_row  = a(seq(1, 7), span).row(5);
    array out_rows = a(seq(1, 7), span).rows(5, 6);
    array out_col  = a(seq(1, 7), span).col(5);
    array out_cols = a(seq(1, 7), span).cols(5, 6);

    ASSERT_EQ(0, where(brow != out_row).elements());
    ASSERT_EQ(0, where(brows != out_rows).elements());
    ASSERT_EQ(0, where(bcol != out_col).elements());
    ASSERT_EQ(0, where(bcols != out_cols).elements());

    array avol    = range(dim4(10, 10, 10));
    array bvol    = avol(seq(1, 7), span, span);
    array bslice  = bvol.slice(5);
    array bslices = bvol.slices(5, 6);

    array out_slice  = avol(seq(1, 7), span, span).slice(5);
    array out_slices = avol(seq(1, 7), span, span).slices(5, 6);

    ASSERT_EQ(0, where(bslice != out_slice).elements());
    ASSERT_EQ(0, where(bslices != out_slices).elements());
}
#endif

TEST(Indexing, SNIPPET_indexing_first) {
    //! [ex_indexing_first]
    array A = array(seq(1, 9), 3, 3);
    af_print(A);
    // 1.0000 4.0000 7.0000
    // 2.0000 5.0000 8.0000
    // 3.0000 6.0000 9.0000

    af_print(A(0));  // first element
    // 1.0000

    af_print(A(0, 1));  // first row, second column
    // 4.0000

    af_print(A(end));  // last element
    // 9.0000

    af_print(A(-1));  // also last element
    // 9.0000

    af_print(A(end - 1));  // second-to-last element
    // 8.0000

    af_print(A(1, span));  // second row
    // 2.0000     5.0000     8.0000

    af_print(A.row(end));  // last row
    // 3.0000     6.0000     9.0000

    af_print(A.cols(1, end));  // all but first column
    // 4.0000     7.0000
    // 5.0000     8.0000
    // 6.0000     9.0000

    float b_host[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    array b(10, 1, b_host);
    af_print(b(seq(3)));
    // 0.0000
    // 1.0000
    // 2.0000

    af_print(b(seq(1, 7)));
    // 1.0000
    // 2.0000
    // 3.0000
    // 4.0000
    // 5.0000
    // 6.0000
    // 7.0000

    af_print(b(seq(1, 7, 2)));
    // 1.0000
    // 3.0000
    // 5.0000
    // 7.0000

    af_print(b(seq(0, end, 2)));
    // 0.0000
    // 2.0000
    // 4.0000
    // 6.0000
    // 8.0000
    //! [ex_indexing_first]

    array lin_first    = A(0);
    array lin_last     = A(end);
    array lin_snd_last = A(end - 1);

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
        for (unsigned i = 0; i < hout.size(); i++) {
            ASSERT_FLOAT_EQ(b_host[i], hout[i]);
        }
    }

    {
        array out = b(seq(1, 7));
        ASSERT_EQ(7, out.elements());
        vector<float> hout(out.elements());
        out.host(&hout.front());
        for (unsigned i = 1; i < hout.size(); i++) {
            ASSERT_FLOAT_EQ(b_host[i], hout[i - 1]);
        }
    }

    {
        array out = b(seq(1, 7, 2));
        ASSERT_EQ(4, out.elements());
        vector<float> hout(out.elements());
        out.host(&hout.front());
        for (unsigned i = 0; i < hout.size(); i++) {
            ASSERT_FLOAT_EQ(b_host[i * 2 + 1], hout[i]);
        }
    }
}

TEST(Indexing, SNIPPET_indexing_set) {
    //! [ex_indexing_set]
    array A = constant(0, 3, 3);
    af_print(A);
    // 0.0000     0.0000     0.0000
    // 0.0000     0.0000     0.0000
    // 0.0000     0.0000     0.0000

    // setting entries to a constant
    A(span) = 4;  // fill entire array
    af_print(A);
    // 4.0000     4.0000     4.0000
    // 4.0000     4.0000     4.0000
    // 4.0000     4.0000     4.0000

    A.row(0) = -1;  // first row
    af_print(A);
    // -1.0000    -1.0000    -1.0000
    //  4.0000     4.0000     4.0000
    //  4.0000     4.0000     4.0000

    A(seq(3)) = 3.1415;  // first three elements
    af_print(A);
    // 3.1415    -1.0000    -1.0000
    // 3.1415     4.0000     4.0000
    // 3.1415     4.0000     4.0000

    // copy in another matrix
    array B = constant(1, 4, 4, s32);
    af_print(B);
    //          1          1          1          1
    //          1          1          1          1
    //          1          1          1          1
    //          1          1          1          1

    B.row(0) = randu(1, 4, f32);  // set a row to random values (also upcast)

    // The first rows are zeros because randu returns values from 0.0 - 1.0
    // and they were converted to the type of B which is s32
    af_print(B);
    //          0          0          0          0
    //          1          1          1          1
    //          1          1          1          1
    //          1          1          1          1
    //! [ex_indexing_set]
    // TODO: Confirm the outputs are correct. see #697
}

TEST(Indexing, SNIPPET_indexing_ref) {
    //! [ex_indexing_ref]
    float h_inds[] = {0, 4, 2, 1};  // zero-based indexing
    array inds(1, 4, h_inds);
    af_print(inds);
    // 0.0000     4.0000     2.0000     1.0000

    array B = randu(1, 4);
    af_print(B);
    // 0.5471     0.3114     0.5535     0.3800

    array c = B(inds);  // get
    af_print(c);
    // 0.5471     0.3800     0.5535     0.3114

    B(inds) = -1;              // set to scalar
    B(inds) = constant(0, 4);  // zero indices
    af_print(B);
    // 0.0000     0.0000     0.0000     0.0000
    //! [ex_indexing_ref]
    // TODO: Confirm the outputs are correct. see #697
}

TEST(Indexing, IndexingCopy) {
    array A = constant(0, 1, s32);
    af::index s1;
    s1 = af::index(A);
    // At exit both A and s1 will be destroyed
    // but the underlying array should only be
    // freed once.
}

TEST(Assign, LinearIndexSeq) {
    const int nx = 5;
    const int ny = 4;

    const int st  = nx - 2;
    const int en  = nx * (ny - 1);
    const int num = (en - st + 1);

    array a       = randu(nx, ny);
    af::index idx = seq(st, en);

    af_array in_arr = a.get();
    af_index_t ii   = idx.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_index(&out_arr, in_arr, 1, &ii.idx.seq));

    array out(out_arr);

    ASSERT_EQ(out.dims(0), num);
    ASSERT_EQ(out.elements(), num);

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < num; i++) { ASSERT_EQ(ha[i + st], hout[i]); }
}

TEST(Assign, LinearIndexGenSeq) {
    const int nx = 5;
    const int ny = 4;

    const int st  = nx - 2;
    const int en  = nx * (ny - 1);
    const int num = (en - st + 1);

    array a       = randu(nx, ny);
    af::index idx = seq(st, en);

    af_array in_arr = a.get();
    af_index_t ii   = idx.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_index_gen(&out_arr, in_arr, 1, &ii));

    array out(out_arr);

    ASSERT_EQ(out.dims(0), num);
    ASSERT_EQ(out.elements(), num);

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < num; i++) { ASSERT_EQ(ha[i + st], hout[i]); }
}

TEST(Assign, LinearIndexGenArr) {
    const int nx = 5;
    const int ny = 4;

    const int st  = nx - 2;
    const int en  = nx * (ny - 1);
    const int num = (en - st + 1);

    array a       = randu(nx, ny);
    af::index idx = array(seq(st, en));

    af_array in_arr = a.get();
    af_index_t ii   = idx.get();
    af_array out_arr;

    ASSERT_SUCCESS(af_index_gen(&out_arr, in_arr, 1, &ii));

    array out(out_arr);

    ASSERT_EQ(out.dims(0), num);
    ASSERT_EQ(out.elements(), num);

    vector<float> hout(nx * ny);
    vector<float> ha(nx * ny);

    a.host(&ha[0]);
    out.host(&hout[0]);

    for (int i = 0; i < num; i++) { ASSERT_EQ(ha[i + st], hout[i]); }
}

TEST(Index, OutOfBounds) {
    uint gold[7]  = {0, 9, 49, 119, 149, 149, 148};
    uint h_idx[7] = {0, 9, 49, 119, 149, 150, 151};
    uint output[7];

    array a = iota(dim4(50, 1, 3)).as(s32);
    array idx(7, h_idx);
    array b = a(idx);
    b.host((void *)output);

    for (int i = 0; i < 7; ++i) ASSERT_EQ(gold[i], output[i]);
}

TEST(Index, ISSUE_1101_FULL) {
    deviceGC();
    array a = randu(5, 5);

    size_t aby, abu, lby, lbu;
    deviceMemInfo(&aby, &abu, &lby, &lbu);

    array b = a(span, span);

    size_t aby1, abu1, lby1, lbu1;
    deviceMemInfo(&aby1, &abu1, &lby1, &lbu1);

    ASSERT_EQ(aby, aby1);
    ASSERT_EQ(abu, abu1);
    ASSERT_EQ(lby, lby1);
    ASSERT_EQ(lbu, lbu1);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST(Index, ISSUE_1101_COL0) {
    deviceGC();
    array a = randu(5, 5);
    vector<float> ha(a.elements());
    a.host(ha.data());
    vector<float> gold(ha.begin(), ha.begin() + 5);

    size_t aby, abu, lby, lbu;
    deviceMemInfo(&aby, &abu, &lby, &lbu);

    array b = a(span, 0);

    size_t aby1, abu1, lby1, lbu1;
    deviceMemInfo(&aby1, &abu1, &lby1, &lbu1);

    ASSERT_EQ(aby, aby1) << "Number of bytes different";
    ASSERT_EQ(abu, abu1) << "Number of buffers different";
    ASSERT_EQ(lby, lby1) << "Number of bytes different";
    ASSERT_EQ(lbu, lbu1) << "Number of buffers different";

    ASSERT_VEC_ARRAY_EQ(gold, dim4(a.dims()[0]), b);
}

TEST(Index, ISSUE_1101_MODDIMS) {
    deviceGC();
    array a = randu(5, 5);
    vector<float> ha(a.elements());
    a.host(&ha[0]);

    size_t aby, abu, lby, lbu;
    deviceMemInfo(&aby, &abu, &lby, &lbu);

    int st  = 0;
    int en  = 9;
    int nx  = 2;
    int ny  = 5;
    array b = a(seq(st, en));
    array c = moddims(b, nx, ny);
    size_t aby1, abu1, lby1, lbu1;
    deviceMemInfo(&aby1, &abu1, &lby1, &lbu1);

    EXPECT_EQ(aby, aby1) << "Number of bytes different";
    EXPECT_EQ(abu, abu1) << "Number of buffers different";
    EXPECT_EQ(lby, lby1) << "Number of bytes different";
    EXPECT_EQ(lbu, lbu1) << "Number of buffers different";

    vector<float> hb(b.elements());
    b.host(&hb[0]);
    for (int i = 0; i < b.elements(); i++) { ASSERT_EQ(ha[i + st], hb[i]); }

    vector<float> hc(c.elements());
    c.host(&hc[0]);
    for (int i = 0; i < c.elements(); i++) { ASSERT_EQ(ha[i + st], hc[i]); }
}

TEST(Index, Issue1846IndexStepCascade) {
    array a = randu(3, 12);
    array b = a(span, seq(0, end, 2));
    array c = b(span, seq(0, end, 3));
    array d = a(span, seq(0, end, 6));
    EXPECT_EQ(allTrue<bool>(c == d), true);
}

TEST(Index, Issue1845IndexStepReorder) {
    array a = randu(1, 8, 1);
    array b = reorder(a, 0, 2, 1);
    array d = reorder(b(0, 0, span), 2, 1, 0);
    EXPECT_EQ(allTrue<bool>(a.T() == d), true);
}

TEST(Index, Issue1867ChainedIndexingLeak) {
    using af::randn;
    using af::sync;
    {
        array lInput = randn(100, 100, f32);
        array Q3     = lInput.rows(0, 3).cols(0, 3);
        Q3.eval();
        sync();
    }
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    ASSERT_EQ(0u, lock_buffers);
}

TEST(Index, InvalidSequence_SingleElementNegativeStep) {
    EXPECT_THROW(af::seq(1, 1, -1), af::exception);
}
TEST(Index, InvalidSequence_PositiveRangeNegativeStep) {
    EXPECT_THROW(af::seq(1, 5, -1), af::exception);
}

TEST(Index, InvalidSequence_NegativeRangePositiveStep) {
    EXPECT_THROW(af::seq(-1, -5, 1), af::exception);
}

TEST(Index, ISSUE_2273) {
    int h_idx[2] = {1, 1};
    array idx(2, h_idx);

    float h_input[12] = {0.f, 1.f, 2.f, 3.f, 4.f,  5.f,
                         6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    array input(2, 3, 2, h_input);
    array input_reord = reorder(input, 0, 2, 1);
    array output      = input_reord(span, idx, span);

    float h_gold[12] = {6.f, 7.f, 6.f,  7.f,  8.f,  9.f,
                        8.f, 9.f, 10.f, 11.f, 10.f, 11.f};
    array gold(2, 2, 3, h_gold);

    ASSERT_ARRAYS_EQ(gold, output);
}

TEST(Index, ISSUE_2273_Flipped) {
    int h_idx[2] = {1, 1};
    array idx(2, h_idx);

    float h_input[12] = {0.f, 1.f, 6.f, 7.f, 2.f,  3.f,
                         8.f, 9.f, 4.f, 5.f, 10.f, 11.f};
    array input(2, 2, 3, h_input);
    array input_reord = reorder(input, 0, 2, 1);
    array input_slice = input_reord(span, span, idx);

    array input_ref       = iota(dim4(2, 3, 2));
    array input_ref_slice = input_ref(span, span, idx);

    float h_gold[12] = {6.f, 7.f, 8.f, 9.f, 10.f, 11.f,
                        6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
    array input_slice_gold(2, 3, 2, h_gold);

    ASSERT_ARRAYS_EQ(input_slice_gold, input_slice);
}

TEST(Index, CopiedIndexDestroyed) {
    array in = randu(10, 10);
    array a  = constant(1, 10);

    af::index index1(a);
    af::index index2(seq(10));

    af::index index3(index1);
    { af::index index4(index1); }

    af_print(in(index1, index2));
}

// clang-format off
class IndexDocs : public ::testing::Test {
public:
  array A;

  void SetUp() {
    //![index_tutorial_1]
    float data[] = {0,  1,  2,  3,
                    4,  5,  6,  7,
                    8,  9, 10, 11,
                   12, 13, 14, 15};
    af::array A(4, 4, data);
    //![index_tutorial_1]
    this->A = A;
  }
};

TEST_F(IndexDocs, Precondition) {
  vector<float> gold(4*4);
  std::iota(gold.begin(), gold.end(), 0.f);
  ASSERT_VEC_ARRAY_EQ(gold, dim4(4, 4), A);
}

TEST_F(IndexDocs, 2_3Element) {
    array out =
    //![index_tutorial_first_element]
    // Returns an array pointing to the first element
    A(2, 3); // WARN: avoid doing this. Demo only
    //![index_tutorial_first_element]
    vector<float> gold(1, 14.f);
    ASSERT_VEC_ARRAY_EQ(gold, dim4(1), out);
}

TEST_F(IndexDocs, FifthElement) {
    array out =
    //![index_tutorial_fifth_element]
    // Returns an array pointing to the fifth element
    A(5);
    //![index_tutorial_fifth_element]
    vector<float> gold(1, 5.f);
    ASSERT_VEC_ARRAY_EQ(gold, dim4(1), out);
}

TEST_F(IndexDocs, NegativeIndexing) {
    //![index_tutorial_negative_indexing]
    array ref0 = A(2, -1);    // 14 second row last column
    array ref1 = A(2, end);   // 14 Same as above
    array ref2 = A(2, -2);    // 10 Second row, second to last(third) column
    array ref3 = A(2, end-1); // 10 Same as above
    //![index_tutorial_negative_indexing]
    vector<float> gold1(1, 14.f);
    vector<float> gold2(1, 10.f);
    ASSERT_VEC_ARRAY_EQ(gold1, dim4(1), ref0);
    ASSERT_VEC_ARRAY_EQ(gold1, dim4(1), ref1);
    ASSERT_VEC_ARRAY_EQ(gold2, dim4(1), ref2);
    ASSERT_VEC_ARRAY_EQ(gold2, dim4(1), ref3);
}

TEST_F(IndexDocs, ThirdColumn) {
    array out =
    //![index_tutorial_third_column]
    // Returns an array pointing to the third column
    A(span, 2);
    //![index_tutorial_third_column]
    vector<float> gold{8, 9, 10, 11};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(4), out);
}

TEST_F(IndexDocs, SecondRow) {
    array out =
    //![index_tutorial_second_row]
    // Returns an array pointing to the second row
    A(1, span);
    //![index_tutorial_second_row]
    vector<float> gold{1, 5, 9, 13};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(1, 4), out);
}

TEST_F(IndexDocs, FirstTwoColumns) {
    array out =
    //![index_tutorial_first_two_columns]
    // Returns an array pointing to the first two columns
    A(span, seq(2));
    //![index_tutorial_first_two_columns]
    vector<float> gold{0, 1, 2, 3, 4, 5, 6, 7};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(4, 2), out);
}

TEST_F(IndexDocs, SecondAndFourthRows) {
    array out =
    //![index_tutorial_second_and_fourth_rows]
    // Returns an array pointing to the second and fourth rows
    A(seq(1, end, 2), span);
    //![index_tutorial_second_and_fourth_rows]
    vector<float> gold{1, 3, 5, 7, 9, 11, 13, 15};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(2, 4), out);
}


TEST_F(IndexDocs, Arrays) {
    //![index_tutorial_array_indexing]
    vector<int> hidx = {2, 1, 3};
    vector<int> hidy = {3, 1, 2};
    array idx(3, hidx.data());
    array idy(3, hidy.data());

    array out = A(idx, idy);
    //![index_tutorial_array_indexing]

    vector<float> gold{
   14.f,    13.f,    15.f,
    6.f,     5.f,     7.f,
   10.f,     9.f,    11.f};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(3, 3), out);
}


TEST_F(IndexDocs, Approx) {
    //![index_tutorial_approx]
    vector<float> hidx = {2, 1, 3};
    vector<float> hidy = {3, 1, 2};
    array idx(3, hidx.data());
    array idy(3, hidy.data());

    array out = approx2(A, idx, idy);
    //![index_tutorial_approx]

    vector<float> gold{14.f, 5.f, 11.f};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(3), out);
}

TEST_F(IndexDocs, Boolean) {
    //![index_tutorial_boolean]
    array out = A(A < 5);
    //![index_tutorial_boolean]
    vector<float> gold = {0, 1, 2, 3, 4};
    ASSERT_VEC_ARRAY_EQ(gold, dim4(5), out);
}

TEST_F(IndexDocs, References) {
    deviceGC();
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    //![index_tutorial_references]
    array reference = A(span, 1);
    array reference2 = A(seq(3), 1);
    array reference3 = A(seq(2), span);
    //![index_tutorial_references]

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    ASSERT_EQ(0, lock_buffers2 - lock_buffers);
}

TEST_F(IndexDocs, Copies) {
    deviceGC();
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    //![index_tutorial_copies]
    array copy = A(2, span);
    array copy2 = A(seq(1, 3, 2), span);


    int hidx[] = {0, 1, 2};
    array idx(3, hidx);
    array copy3 = A(idx, span);
    //![index_tutorial_copies]

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    ASSERT_EQ(3, lock_buffers2 - lock_buffers);
}

TEST_F(IndexDocs, Assignment) {
    deviceGC();
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    //![index_tutorial_assignment]
    array inputA = constant(3, 10, 10);
    array inputB = constant(2, 10, 10);
    array data   = constant(1, 10, 10);

    // Points to the second column of data. Does not allocate memory
    array ref = data(span, 1);

    // This call does NOT update data. Memory allocated in matmul
    ref = matmul(inputA, inputB);
    // reference does not point to the same memory as the data array
    //![index_tutorial_assignment]

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    vector<float> gold_reference(100, 60);
    vector<float> gold_data(100, 1);
    ASSERT_VEC_ARRAY_EQ(gold_reference, dim4(10, 10), ref);
    ASSERT_VEC_ARRAY_EQ(gold_data, dim4(10, 10), data);
    ASSERT_EQ(4, lock_buffers2 - lock_buffers);
}

TEST_F(IndexDocs, AssignmentThirdColumn) {
    vector<float> gold(A.elements());
    A.host(gold.data());

    deviceGC();
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    //![index_tutorial_assignment_third_column]
    array reference = A(span, 2);
    A(span, 2) = 3.14f;
    assert(allTrue<bool>(reference != A(span, 2)));
    //![index_tutorial_assignment_third_column]
    vector<float> gold_reference(begin(gold) + 8, begin(gold)+12);
    ASSERT_VEC_ARRAY_EQ(gold_reference, dim4(4), reference);
    gold[8] = gold[9] = gold[10] = gold[11] = 3.14f;
    ASSERT_VEC_ARRAY_EQ(gold, A.dims(), A);

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    ASSERT_EQ(1, lock_buffers2 - lock_buffers);
}

TEST_F(IndexDocs, AssignmentAlloc) {
    deviceGC();
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    //![index_tutorial_assignment_alloc]
    {
        // No allocation performed. ref points to A's memory
        array ref = A(span, 2);
    } // ref goes out of scope. No one point's to A's memory
    A(span, 2) = 3.14f; // No allocation performed.
    //![index_tutorial_assignment_alloc]

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    ASSERT_EQ(0, lock_buffers2 - lock_buffers);
}

TEST_F(IndexDocs, AssignmentRaceCondition) {
    //![index_tutorial_assignment_race_condition]
    vector<int> hidx = {4, 3, 4, 0};
    vector<float> hvals = {9.f, 8.f, 7.f, 6.f};
    array idx(4, hidx.data());
    array vals(4, hvals.data());

    A(idx) = vals; // nondeterministic. A(4) can be 9 or 7
    //![index_tutorial_assignment_race_condition]
}

// clang-format on
