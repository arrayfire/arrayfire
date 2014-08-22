#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <af/reduce.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::af_cfloat;
using af::af_cdouble;

typedef af_err (*reduceFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, reduceFunc af_reduce>
void reduceTest(string pTestFile, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    af::dim4            dims(1);
    vector<int>         data;
    vector<vector<int>> tests;
    ReadTests<int, int> (pTestFile,dims,data,tests);

    vector<Ti> in(begin(data), end(data));

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<To> currGoldBar(begin(tests[d]), end(tests[d]));

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_reduce(&outArray, inArray, d));

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for dim " << d << std::endl;
        }

        // Delete
        delete[] outData;
    }


    if(inArray   != 0) af_destroy_array(inArray);
    if(outArray  != 0) af_destroy_array(outArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

vector<af_seq> init_subs()
{
    vector<af_seq> subs;
    subs.push_back({2, 6, 1});
    subs.push_back({1, 5, 1});
    subs.push_back({1, 3, 1});
    subs.push_back({1, 2, 1});
    return subs;
}

#define REDUCE_TESTS(FN, TAG, Ti, To)                   \
    TEST(Reduce,Test_##FN##_##TAG)                      \
    {                                                   \
        reduceTest<Ti, To, af_##FN>(                    \
            string(TEST_DIR"/reduce/"#FN".test")        \
            );                                          \
    }                                                   \
    TEST(Reduce,Test_##FN##_subs_##TAG)                 \
    {                                                   \
        reduceTest<Ti, To, af_##FN>(                    \
            string(TEST_DIR"/reduce/"#FN"Subs.test"),   \
            true, init_subs());                         \
    }                                                   \

REDUCE_TESTS(sum, float   , float     , float     );
REDUCE_TESTS(sum, double  , double    , double    );
REDUCE_TESTS(sum, int     , int       , int       );
REDUCE_TESTS(sum, cfloat  , af_cfloat , af_cfloat );
REDUCE_TESTS(sum, cdouble , af_cdouble, af_cdouble);
REDUCE_TESTS(sum, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(sum, uchar   , unsigned char, unsigned);

REDUCE_TESTS(min, float   , float     , float     );
REDUCE_TESTS(min, double  , double    , double    );
REDUCE_TESTS(min, int     , int       , int       );
REDUCE_TESTS(min, cfloat  , af_cfloat , af_cfloat );
REDUCE_TESTS(min, cdouble , af_cdouble, af_cdouble);
REDUCE_TESTS(min, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(min, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(max, float   , float     , float     );
REDUCE_TESTS(max, double  , double    , double    );
REDUCE_TESTS(max, int     , int       , int       );
REDUCE_TESTS(max, cfloat  , af_cfloat , af_cfloat );
REDUCE_TESTS(max, cdouble , af_cdouble, af_cdouble);
REDUCE_TESTS(max, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(max, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(anytrue, float   , float     , unsigned char);
REDUCE_TESTS(anytrue, double  , double    , unsigned char);
REDUCE_TESTS(anytrue, int     , int       , unsigned char);
REDUCE_TESTS(anytrue, cfloat  , af_cfloat , unsigned char);
REDUCE_TESTS(anytrue, cdouble , af_cdouble, unsigned char);
REDUCE_TESTS(anytrue, unsigned, unsigned  , unsigned char);
REDUCE_TESTS(anytrue, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(alltrue, float   , float     , unsigned char);
REDUCE_TESTS(alltrue, double  , double    , unsigned char);
REDUCE_TESTS(alltrue, int     , int       , unsigned char);
REDUCE_TESTS(alltrue, cfloat  , af_cfloat , unsigned char);
REDUCE_TESTS(alltrue, cdouble , af_cdouble, unsigned char);
REDUCE_TESTS(alltrue, unsigned, unsigned  , unsigned char);
REDUCE_TESTS(alltrue, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(count, float   , float     , unsigned);
REDUCE_TESTS(count, double  , double    , unsigned);
REDUCE_TESTS(count, int     , int       , unsigned);
REDUCE_TESTS(count, cfloat  , af_cfloat , unsigned);
REDUCE_TESTS(count, cdouble , af_cdouble, unsigned);
REDUCE_TESTS(count, unsigned, unsigned  , unsigned);
REDUCE_TESTS(count, uchar   , unsigned char, unsigned);
