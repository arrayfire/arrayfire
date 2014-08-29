#include <Array.hpp>

namespace cpu
{

template<typename T, bool isColor>
Array<T> * bilateral(const Array<T> &in, const float &s_sigma, const float &c_sigma);

}
