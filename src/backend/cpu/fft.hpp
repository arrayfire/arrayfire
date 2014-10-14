#include <Array.hpp>

namespace cpu
{

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double normalize, dim_type const npad, dim_type const * const pad);

template<typename T, int rank>
Array<T> * ifft(Array<T> const &in, double normalize, dim_type const npad, dim_type const * const pad);

}
