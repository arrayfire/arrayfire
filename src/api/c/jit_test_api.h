/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

#ifdef __cplusplus
namespace af {
/// Get the maximum jit tree length for active backend
///
/// \returns the maximum length of jit tree from root to any leaf
AFAPI int getMaxJitLen(void);

/// Set the maximum jit tree length for active backend
///
/// \param[in] jit_len is the maximum length of jit tree from root to any
/// leaf
AFAPI void setMaxJitLen(const int jitLen);
}  // namespace af
#endif  //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/// Get the maximum jit tree length for active backend
///
/// \param[out] jit_len is the maximum length of jit tree from root to any
/// leaf
///
/// \returns Always returns AF_SUCCESS
AFAPI af_err af_get_max_jit_len(int *jit_len);

/// Set the maximum jit tree length for active backend
///
/// \param[in] jit_len is the maximum length of jit tree from root to any
/// leaf
///
/// \returns Always returns AF_SUCCESS
AFAPI af_err af_set_max_jit_len(const int jit_len);

#ifdef __cplusplus
}
#endif
