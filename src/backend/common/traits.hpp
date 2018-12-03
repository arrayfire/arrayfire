/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/defines.h>

namespace af {
template<typename T>
struct dtype_traits;
}

namespace common {
class half;
}

namespace af {
template<>
struct dtype_traits<common::half> {
    enum { af_type = f16, ctype = f16 };
    typedef common::half base_type;
    static const char* getName() { return "half"; }
};
}  // namespace af
