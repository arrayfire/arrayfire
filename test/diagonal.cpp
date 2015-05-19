
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <arrayfire.h>
#include <iostream>

using namespace std;
using namespace af;

template<typename T>
class Diagonal : public ::testing::Test
{

};

typedef ::testing::Types<float, double, int, uint, char, unsigned char> TestTypes;
TYPED_TEST_CASE(Diagonal, TestTypes);

TYPED_TEST(Diagonal, Create)
{
    if (noDoubleTests<TypeParam>()) return;
    try {

        static const int size = 1000;
        vector<TypeParam> input (size * size);
        for(int i = 0; i < size; i++) {
            input[i] = i;
        }
        for(int jj = 10; jj < size; jj+=100) {
            array data(jj, &input.front(), af::afHost, dtype_traits<TypeParam>::af_type);
            array out = diag(data, 0, false);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for(int i =0; i < (int)out.dims(0); i++) {
                for(int j =0; j < (int)out.dims(1); j++) {
                    if(i == j) ASSERT_EQ(input[i], h_out[i * out.dims(0) + j]);
                    else       ASSERT_EQ(TypeParam(0), h_out[i * out.dims(0) + j]);
                }
            }
        }
    } catch (const af::exception& ex) {
        FAIL() << ex.what() << endl;
    }
}

TYPED_TEST(Diagonal, Extract)
{
    if (noDoubleTests<TypeParam>()) return;

    try {
        static const int size = 1000;
        vector<TypeParam> input (size * size);
        for(int i = 0; i < size * size; i++) {
            input[i] = i;
        }
        for(int jj = 10; jj < size; jj+=100) {
            array data(jj, jj, &input.front(), af::afHost, dtype_traits<TypeParam>::af_type);
            array out = diag(data, 0);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for(int i =0; i < (int)out.dims(0); i++) {
                ASSERT_EQ(input[i * data.dims(0) + i], h_out[i]);
            }
        }
    } catch (const af::exception& ex) {
        FAIL() << ex.what() << endl;
    }
}
