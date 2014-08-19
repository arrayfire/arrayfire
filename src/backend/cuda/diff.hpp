#include <af/array.h>

namespace cuda
{
    template<typename T>
    af_array diff1(const af_array &in, const int dim);

    template<typename T>
    af_array diff2(const af_array &in, const int dim);
}
