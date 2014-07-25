
#include <af/array.h>
#include <af/defines.h>


namespace opencl
{

template<typename T>
void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index);

}
