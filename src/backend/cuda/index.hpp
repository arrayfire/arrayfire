
#include <af/array.h>
#include <af/dim4.hpp>
#include <vector>


namespace cuda
{

template<typename T>
void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);

}
