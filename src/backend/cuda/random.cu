#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <random.hpp>
#include <cassert>

namespace cuda
{
    template<typename T>
    Array<T>* randu(const af::dim4 &dims)
    {
        assert(1!=1);
        return NULL;
    }

    template<typename T>
    Array<T>* randn(const af::dim4 &dims)
    {
        assert(1!=1);
        return NULL;
    }

    template Array<float>  * randu<float>   (const af::dim4 &dims);
    template Array<double> * randu<double>  (const af::dim4 &dims);
    template Array<cfloat> * randu<cfloat>  (const af::dim4 &dims);
    template Array<cdouble>* randu<cdouble> (const af::dim4 &dims);
    template Array<int>    * randu<int>     (const af::dim4 &dims);
    template Array<uint>   * randu<uint>    (const af::dim4 &dims);
    template Array<char>   * randu<char>    (const af::dim4 &dims);
    template Array<uchar>  * randu<uchar>   (const af::dim4 &dims);

    template Array<float>  * randn<float>   (const af::dim4 &dims);
    template Array<double> * randn<double>  (const af::dim4 &dims);
    template Array<cfloat> * randn<cfloat>  (const af::dim4 &dims);
    template Array<cdouble>* randn<cdouble> (const af::dim4 &dims);

}
