#include "gtest/gtest.h"
#include "arrayfire.h"
#include <af/dim4.hpp>
#include <af/defines.h>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <string>

using std::vector;
using std::string;
using std::generate;
using std::cout;
using std::endl;
using std::begin;
using std::end;

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

    long d[1] = {dims};

    vector<T> hData(dims);
    T n(0);
    generate(begin(hData), end(hData), [&] () { return n++; });

    af_array a = 0;
    EXPECT_EQ(AF_SUCCESS, af_create_array(&a, ndims, d, (af_dtype) af::dtype_traits<T>::af_type));
    EXPECT_EQ(AF_SUCCESS, af_copy(&a, &hData.front()));

    vector<af_array> indexed_array(seqs.size(), 0);
    for(size_t i = 0; i < seqs.size(); i++) {
        EXPECT_EQ(AF_SUCCESS, af_index(&(indexed_array[i]), a, ndims, &seqs[i]))
        << "where seqs[i].begin == "    << seqs[i].begin
        << " seqs[i].step == "          << seqs[i].step
        << " seqs[i].end == "           << seqs[i].end;
    }

    vector<T*> h_indexed(seqs.size());
    for(size_t i = 0; i < seqs.size(); i++) {
        //af_print(indexed_array[i]);
        EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&(h_indexed[i]), indexed_array[i]));
    }

    for(size_t k = 0; k < seqs.size(); k++) {
        if(seqs[k].step > 0)        {
            checkValues(seqs[k], &hData.front(), h_indexed[k], std::less_equal<int>());
        }
        else if (seqs[k].step < 0)  {
            checkValues(seqs[k], &hData.front(), h_indexed[k], std::greater_equal<int>());
        } //reverse indexing
        else                        {                                                 //span
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

using std::array;

template<typename T>
class Indexing2D : public ::testing::Test
{
public:
    virtual void SetUp() {

        column_continuous_seq.push_back({{span, {  0,  6,  1}}});
        column_continuous_seq.push_back({{span, {  4,  9,  1}}});
        column_continuous_seq.push_back({{span, {  3,  8,  1}}});

        column_continuous_reverse_seq.push_back({{span, {  6,  0,  -1}}});
        column_continuous_reverse_seq.push_back({{span, {  9,  4,  -1}}});
        column_continuous_reverse_seq.push_back({{span, {  8,  3,  -1}}});

        column_strided_seq.push_back({{span, {  0,    8,   2 }}}); // Two Step
        column_strided_seq.push_back({{span, {  2,    9,   3 }}}); // Three Step
        column_strided_seq.push_back({{span, {  0,    9,   4 }}}); // Four Step

        column_strided_reverse_seq.push_back({{span, {  8,   0,   -2 }}}); // Two Step
        column_strided_reverse_seq.push_back({{span, {  9,   2,   -3 }}}); // Three Step
        column_strided_reverse_seq.push_back({{span, {  9,   0,   -4 }}}); // Four Step

        row_continuous_seq.push_back({{{  0,  6,  1}, span}});
        row_continuous_seq.push_back({{{  4,  9,  1}, span}});
        row_continuous_seq.push_back({{{  3,  8,  1}, span}});

        row_continuous_reverse_seq.push_back({{{  6,  0,  -1}, span}});
        row_continuous_reverse_seq.push_back({{{  9,  4,  -1}, span}});
        row_continuous_reverse_seq.push_back({{{  8,  3,  -1}, span}});

        row_strided_seq.push_back({{{  0,    8,   2 }, span}});
        row_strided_seq.push_back({{{  2,    9,   3 }, span}});
        row_strided_seq.push_back({{{  0,    9,   4 }, span}});

        row_strided_reverse_seq.push_back({{{  8,   0,   -2 }, span}});
        row_strided_reverse_seq.push_back({{{  9,   2,   -3 }, span}});
        row_strided_reverse_seq.push_back({{{  9,   0,   -4 }, span}});

        continuous_continuous_seq.push_back({{{  1,  6,  1}, {  0,  6,  1}}});
        continuous_continuous_seq.push_back({{{  3,  9,  1}, {  4,  9,  1}}});
        continuous_continuous_seq.push_back({{{  5,  8,  1}, {  3,  8,  1}}});

        continuous_reverse_seq.push_back({{{  1,  6,  1}, {  6,  0,  -1}}});
        continuous_reverse_seq.push_back({{{  3,  9,  1}, {  9,  4,  -1}}});
        continuous_reverse_seq.push_back({{{  5,  8,  1}, {  8,  3,  -1}}});

        continuous_strided_seq.push_back({{{  1,  6,  1}, {  0,  8,  2}}});
        continuous_strided_seq.push_back({{{  3,  9,  1}, {  2,  9,  3}}});
        continuous_strided_seq.push_back({{{  5,  8,  1}, {  1,  9,  4}}});

        continuous_strided_reverse_seq.push_back({{{  1,  6,  1}, {  8,  0,  -2}}});
        continuous_strided_reverse_seq.push_back({{{  3,  9,  1}, {  9,  2,  -3}}});
        continuous_strided_reverse_seq.push_back({{{  5,  8,  1}, {  9,  1,  -4}}});

        reverse_continuous_seq.push_back({{{  6,  1,  -1}, {  0,  6,  1}}});
        reverse_continuous_seq.push_back({{{  9,  3,  -1}, {  4,  9,  1}}});
        reverse_continuous_seq.push_back({{{  8,  5,  -1}, {  3,  8,  1}}});

        reverse_reverse_seq.push_back({{{  6,  1,  -1}, {  6,  0,  -1}}});
        reverse_reverse_seq.push_back({{{  9,  3,  -1}, {  9,  4,  -1}}});
        reverse_reverse_seq.push_back({{{  8,  5,  -1}, {  8,  3,  -1}}});

        reverse_strided_seq.push_back({{{  6,  1,  -1}, {  0,  8,  2}}});
        reverse_strided_seq.push_back({{{  9,  3,  -1}, {  2,  9,  3}}});
        reverse_strided_seq.push_back({{{  8,  5,  -1}, {  1,  9,  4}}});

        reverse_strided_reverse_seq.push_back({{{  6,  1,  -1}, {  8,  0,  -2}}});
        reverse_strided_reverse_seq.push_back({{{  9,  3,  -1}, {  9,  2,  -3}}});
        reverse_strided_reverse_seq.push_back({{{  8,  5,  -1}, {  9,  1,  -4}}});

        strided_continuous_seq.push_back({{{  0,  8,  2}, {  0,  6,  1}}});
        strided_continuous_seq.push_back({{{  2,  9,  3}, {  4,  9,  1}}});
        strided_continuous_seq.push_back({{{  1,  9,  4}, {  3,  8,  1}}});

        strided_strided_seq.push_back({{{  1,  6,  2}, {  0,  8,  2}}});
        strided_strided_seq.push_back({{{  3,  9,  2}, {  2,  9,  3}}});
        strided_strided_seq.push_back({{{  5,  8,  2}, {  1,  9,  4}}});
        strided_strided_seq.push_back({{{  1,  6,  3}, {  0,  8,  2}}});
        strided_strided_seq.push_back({{{  3,  9,  3}, {  2,  9,  3}}});
        strided_strided_seq.push_back({{{  5,  8,  3}, {  1,  9,  4}}});
        strided_strided_seq.push_back({{{  1,  6,  4}, {  0,  8,  2}}});
        strided_strided_seq.push_back({{{  3,  9,  4}, {  2,  9,  3}}});
        strided_strided_seq.push_back({{{  3,  8,  4}, {  1,  9,  4}}});
        strided_strided_seq.push_back({{{  3,  6,  4}, {  1,  9,  4}}});
    }

    vector<std::array<af_seq, 2>> column_continuous_seq;
    vector<std::array<af_seq, 2>> column_continuous_reverse_seq;
    vector<std::array<af_seq, 2>> column_strided_seq;
    vector<std::array<af_seq, 2>> column_strided_reverse_seq;

    vector<std::array<af_seq, 2>> row_continuous_seq;
    vector<std::array<af_seq, 2>> row_continuous_reverse_seq;
    vector<std::array<af_seq, 2>> row_strided_seq;
    vector<std::array<af_seq, 2>> row_strided_reverse_seq;

    vector<std::array<af_seq, 2>> continuous_continuous_seq;
    vector<std::array<af_seq, 2>> continuous_strided_seq;
    vector<std::array<af_seq, 2>> continuous_reverse_seq;
    vector<std::array<af_seq, 2>> continuous_strided_reverse_seq;

    vector<std::array<af_seq, 2>> reverse_continuous_seq;
    vector<std::array<af_seq, 2>> reverse_reverse_seq;
    vector<std::array<af_seq, 2>> reverse_strided_seq;
    vector<std::array<af_seq, 2>> reverse_strided_reverse_seq;

    vector<std::array<af_seq, 2>> strided_continuous_seq;
    vector<std::array<af_seq, 2>> strided_strided_seq;
};

#include <fstream>
#include <iterator>

using std::copy;
using std::istream_iterator;
using std::ostream_iterator;

template<typename InputType, typename ReturnType>
void
ReadTests(const string &FileName, af::dim4 &dims, vector<ReturnType> &out, vector<vector<ReturnType>> &tests) {
    std::ifstream testFile(FileName);
    if(testFile.good()) {
        testFile >> dims;
        vector<InputType>         data(dims.elements());

        unsigned testCount;
        testFile >> testCount;
        tests.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        copy_n( istream_iterator<InputType>(testFile),
                dims.elements(),
                begin(data));

        copy(   begin(data),
                end(data),
                back_inserter(out));

        for(unsigned i = 0; i < testCount; i++) {
            copy_n( istream_iterator<int>(testFile),
                    testSizes[i],
                    back_inserter(tests[i]));
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}

template<typename T, size_t NDims>
void
DimCheck2D(const vector<array<af_seq,NDims>> &seqs,string TestFile)
{
    af::dim4 dimensions(1);
    vector<T> hData;

    vector<vector<T>> tests;
    ReadTests<int, T>(TestFile, dimensions, hData, tests);

    af_array a = 0;
    EXPECT_EQ(AF_SUCCESS, af_create_array(&a, NDims, dimensions.get(), (af_dtype) af::dtype_traits<T>::af_type));
    EXPECT_EQ(AF_SUCCESS, af_copy(&a, &hData.front()));
    //af_print(a); fflush(stdout);

    vector<af_array> indexed_arrays(seqs.size(), 0);
    for(size_t i = 0; i < seqs.size(); i++) {
        EXPECT_EQ(AF_SUCCESS, af_index(&(indexed_arrays[i]), a, NDims, seqs[i].data()));
        //af_print(indexed_arrays[i]);
    }

    vector<T*> h_indexed(seqs.size(), nullptr);
    for(size_t i = 0; i < seqs.size(); i++) {
        //af_print(indexed_arrays[i]);
        EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **) &h_indexed[i], indexed_arrays[i]));
        T* ptr = h_indexed[i];
        if(false == equal(ptr, ptr + tests[i].size(), begin(tests[i]))) {
            af_print(indexed_arrays[i]);
            cout << "index data: ";
            copy(ptr, ptr + tests[i].size(), ostream_iterator<T>(cout, ", "));
            cout << endl << "file data: ";
            copy(begin(tests[i]), end(tests[i]), ostream_iterator<T>(cout, ", "));
            FAIL() << "indexed_array[" << i << "] FAILED" << endl;
        }
    }
}

TYPED_TEST_CASE(Indexing2D, TestTypes);

TYPED_TEST(Indexing2D, ColumnContinious)
{
    DimCheck2D<TypeParam, 2>(this->column_continuous_seq, "./test/data/ColumnContinious.test");
}

TYPED_TEST(Indexing2D, ColumnContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->column_continuous_reverse_seq, "./test/data/ColumnContiniousReverse.test");
}

TYPED_TEST(Indexing2D, ColumnStrided)
{
    DimCheck2D<TypeParam, 2>(this->column_strided_seq, "./test/data/ColumnStrided.test");
}

TYPED_TEST(Indexing2D, ColumnStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->column_strided_reverse_seq, "./test/data/ColumnStridedReverse.test");
}

TYPED_TEST(Indexing2D, RowContinious)
{
    DimCheck2D<TypeParam, 2>(this->row_continuous_seq, "./test/data/RowContinious.test");
}

TYPED_TEST(Indexing2D, RowContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->row_continuous_reverse_seq, "./test/data/RowContiniousReverse.test");
}

TYPED_TEST(Indexing2D, RowStrided)
{
    DimCheck2D<TypeParam, 2>(this->row_strided_seq, "./test/data/RowStrided.test");
}

TYPED_TEST(Indexing2D, RowStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->row_strided_reverse_seq, "./test/data/RowStridedReverse.test");
}

TYPED_TEST(Indexing2D, ContiniousContinious)
{
    DimCheck2D<TypeParam, 2>(this->continuous_continuous_seq, "./test/data/ContiniousContinious.test");
}

TYPED_TEST(Indexing2D, ContiniousReverse)
{
    DimCheck2D<TypeParam, 2>(this->continuous_reverse_seq, "./test/data/ContiniousReverse.test");
}

TYPED_TEST(Indexing2D, ContiniousStrided)
{
    DimCheck2D<TypeParam, 2>(this->continuous_strided_seq, "./test/data/ContiniousStrided.test");
}

TYPED_TEST(Indexing2D, ContiniousStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->continuous_strided_reverse_seq, "./test/data/ContiniousStridedReverse.test");
}

TYPED_TEST(Indexing2D, ReverseContinious)
{
    DimCheck2D<TypeParam, 2>(this->reverse_continuous_seq, "./test/data/ReverseContinious.test");
}

TYPED_TEST(Indexing2D, ReverseReverse)
{
    DimCheck2D<TypeParam, 2>(this->reverse_reverse_seq, "./test/data/ReverseReverse.test");
}

TYPED_TEST(Indexing2D, ReverseStrided)
{
    DimCheck2D<TypeParam, 2>(this->reverse_strided_seq, "./test/data/ReverseStrided.test");
}

TYPED_TEST(Indexing2D, ReverseStridedReverse)
{
    DimCheck2D<TypeParam, 2>(this->reverse_strided_reverse_seq, "./test/data/ReverseStridedReverse.test");
}

TYPED_TEST(Indexing2D, StridedContinious)
{
    DimCheck2D<TypeParam, 2>(this->strided_continuous_seq, "./test/data/StridedContinious.test");
}

TYPED_TEST(Indexing2D, StridedStrided)
{
    DimCheck2D<TypeParam, 2>(this->strided_strided_seq, "./test/data/StridedStrided.test");
}

