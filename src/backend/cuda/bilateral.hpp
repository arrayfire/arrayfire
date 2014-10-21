#include <Array.hpp>

namespace cuda
{

template<typename inType, typename outType, bool isColor>
Array<outType> * bilateral(const Array<inType> &in, const float &s_sigma, const float &c_sigma);

}
