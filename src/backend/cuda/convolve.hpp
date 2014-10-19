#include <Array.hpp>
#include <convolve_common.hpp>

namespace cuda
{

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> * convolve(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);

template<typename T, typename accT, bool expand>
Array<T> * convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

}
