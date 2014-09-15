#include <type_util.hpp>

const char *getName(af_dtype type)
{
    switch(type) {
    case f32: return "float";
    case f64: return "double";
    case c32: return "complex float";
    case c64: return "complex double";
    case u32: return "unsigned int";
    case s32: return "int";
    case u8: return "unsigned char";
    case s8: return "char";
    case b8: return "bool";
    default: return "unknown type";
    }
}
