#include <af/array.h>
#include <Array.hpp>
#include <vector>


namespace cpu
{

template<typename T>
void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);

}
