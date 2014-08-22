#include <type_traits>
#include <random>
#include <limits>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <helper.hpp>
#include <Array.hpp>
#include <random.hpp>

namespace cpu
{

#define RAND(T, D, dist, a, b)                                                              \
    void generate##D(T *outPtr, dim_type nElements)                                         \
    {                                                                                       \
        std::default_random_engine generator;                                               \
        dist<T> distribution(a, b);                                                         \
                                                                                            \
        auto gen = std::bind(distribution, generator);                                      \
        for (int i = 0; i < nElements; ++i) {                                               \
            outPtr[i] = gen();                                                              \
        }                                                                                   \
    }


#define RANDC(T, TY, D, dist, a, b)                                                         \
    void generate##D(T *outPtr, dim_type nElements)                                         \
    {                                                                                       \
        std::default_random_engine generator;                                               \
        dist<TY> distribution(a, b);                                                        \
                                                                                            \
        auto gen = std::bind(distribution, generator);                                      \
        for (int i = 0; i < nElements; ++i) {                                               \
            outPtr[i] = T(gen(), gen());                                                    \
        }                                                                                   \
    }

RAND(float,   U, std::uniform_real_distribution, 0, 1.0);
RAND(double,  U, std::uniform_real_distribution, 0, 1.0);
RAND(int,     U, std::uniform_int_distribution,  0, std::numeric_limits<int>::max());
RAND(uint,    U, std::uniform_int_distribution,  0, std::numeric_limits<uint>::max());
RAND(char,    U, std::uniform_int_distribution,  0, std::numeric_limits<char>::max());
RAND(uchar,   U, std::uniform_int_distribution,  0, std::numeric_limits<uchar>::max());
RANDC(cfloat,  float,  U, std::uniform_real_distribution, 0.0, 1.0);
RANDC(cdouble, double, U, std::uniform_real_distribution, 0.0, 1.0);

RAND(float,   N, std::normal_distribution, 0, 1.0);
RAND(double,  N, std::normal_distribution, 0, 1.0);
RANDC(cfloat,  float,  N, std::normal_distribution, 0.0, 1.0);
RANDC(cdouble, double, N, std::normal_distribution, 0.0, 1.0);

    template<typename T>
    Array<T>* randu(const af::dim4 &dims)
    {
        Array<T> *outArray = createValueArray(dims, (T)0);

        T *outPtr = outArray->get();

        generateU(outPtr, dims.elements());

        return outArray;
    }

    template Array<float>  * randu<float>   (const af::dim4 &dims);
    template Array<double> * randu<double>  (const af::dim4 &dims);
    template Array<cfloat> * randu<cfloat>  (const af::dim4 &dims);
    template Array<cdouble>* randu<cdouble> (const af::dim4 &dims);
    template Array<int>    * randu<int>     (const af::dim4 &dims);
    template Array<uint>   * randu<uint>    (const af::dim4 &dims);
    template Array<char>   * randu<char>    (const af::dim4 &dims);
    template Array<uchar>  * randu<uchar>   (const af::dim4 &dims);

    template<typename T>
    Array<T>* randn(const af::dim4 &dims)
    {
        Array<T> *outArray = createValueArray(dims, (T)0);

        T *outPtr = outArray->get();

        generateN(outPtr, dims.elements());

        return outArray;
    }

    template Array<float>  * randn<float>   (const af::dim4 &dims);
    template Array<double> * randn<double>  (const af::dim4 &dims);
    template Array<cfloat> * randn<cfloat>  (const af::dim4 &dims);
    template Array<cdouble>* randn<cdouble> (const af::dim4 &dims);

}

