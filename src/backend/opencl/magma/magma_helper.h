/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_HELPER_H
#define __MAGMA_HELPER_H

template<typename T> T magma_one();
template<typename T> T magma_neg_one();
template<typename T> magma_int_t magma_get_getrf_nb(int num);

#endif
