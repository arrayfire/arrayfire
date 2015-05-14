#include <af/image.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array regions(const array& in, const af::connectivity connectivity, const af::dtype type)
{
    af_array temp = 0;
    AF_THROW(af_regions(&temp, in.get(), connectivity, type));
    return array(temp);
}

}
