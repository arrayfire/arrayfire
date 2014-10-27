#include <Array.hpp>

namespace cuda
{

template<typename T>
Array<T> * regions(const Array<uchar> &in, af_connectivity_type connectivity);

}
