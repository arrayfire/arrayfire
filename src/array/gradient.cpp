#include <af/image.h>
#include <utility>
#include "error.hpp"

namespace af
{

void grad(array &rows, array &cols, const array& in)
{
    af_array rows_handle = 0;
    af_array cols_handle = 0;
    AF_THROW(af_gradient(&rows_handle, &cols_handle, in.get()));
    rows = array(rows_handle);
    cols = array(cols_handle);
}

}
