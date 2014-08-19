#include <type_traits>
#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <diff.hpp>

namespace opencl
{
    template<typename T>
    af_array diff1(const af_array &in, const int dim)
    {
        assert(1!=1);

        // Create output placeholder
        Array<T> *outArray = createValueArray(af::dim4(1), (T)0);
        return getHandle(*outArray);
    }

    template af_array diff1<float>        (const af_array &in, const int dim);
    template af_array diff1<cfloat>       (const af_array &in, const int dim);
    template af_array diff1<double>       (const af_array &in, const int dim);
    template af_array diff1<cdouble>      (const af_array &in, const int dim);
    template af_array diff1<char>         (const af_array &in, const int dim);
    template af_array diff1<int>          (const af_array &in, const int dim);
    template af_array diff1<unsigned>     (const af_array &in, const int dim);
    template af_array diff1<unsigned char>(const af_array &in, const int dim);

    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    af_array diff2(const af_array &in, const int dim)
    {
        assert(1!=1);

        // Create output placeholder
        Array<T> *outArray = createValueArray(af::dim4(1), (T)0);
        return getHandle(*outArray);
    }

    template af_array diff2<float>        (const af_array &in, const int dim);
    template af_array diff2<cfloat>       (const af_array &in, const int dim);
    template af_array diff2<double>       (const af_array &in, const int dim);
    template af_array diff2<cdouble>      (const af_array &in, const int dim);
    template af_array diff2<char>         (const af_array &in, const int dim);
    template af_array diff2<int>          (const af_array &in, const int dim);
    template af_array diff2<unsigned>     (const af_array &in, const int dim);
    template af_array diff2<unsigned char>(const af_array &in, const int dim);
}
