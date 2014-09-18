#pragma once
#include <ops.hpp>
template<af_op_t T> static const char *binOpName() { return "ADD_OP"; }

template<> const char *binOpName<af_add_t>() { return "ADD_OP"; }
template<> const char *binOpName<af_and_t>() { return "AND_OP"; }
template<> const char *binOpName<af_or_t >() { return "OR_OP" ; }
template<> const char *binOpName<af_min_t>() { return "MIN_OP"; }
template<> const char *binOpName<af_max_t>() { return "MAX_OP"; }
template<> const char *binOpName<af_notzero_t>() { return "NOTZERO_OP"; }
