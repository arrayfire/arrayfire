#include <Array.hpp>

namespace cpu
{

template<typename inType, typename outType>
Array<outType> * histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval);

}
