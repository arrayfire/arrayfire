#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <JIT/BinaryNode.hpp>

namespace opencl
{

    template<typename To, typename Ti, af_op_t op>
    struct BinOp
    {
        const char *name()
        {
            return "noop";
        }
    };

#define BINARY(fn)                              \
    template<typename To, typename Ti>          \
    struct BinOp<To, Ti, af_##fn##_t>           \
    {                                           \
        const char *name()                      \
        {                                       \
            return "__"#fn;                     \
        }                                       \
    };                                          \
                                                \
    template<typename To>                       \
    struct BinOp<To, cfloat, af_##fn##_t>       \
    {                                           \
        const char *name()                      \
        {                                       \
            return "__c"#fn"f";                 \
        }                                       \
    };                                          \
                                                \
    template<typename To>                       \
    struct BinOp<To, cdouble, af_##fn##_t>      \
    {                                           \
        const char *name()                      \
        {                                       \
            return "__c"#fn;                    \
        }                                       \
    };                                          \


BINARY(eq)
BINARY(neq)
BINARY(lt)
BINARY(le)
BINARY(gt)
BINARY(ge)
BINARY(add)
BINARY(sub)
BINARY(mul)
BINARY(div)

#undef BINARY

template<typename To, typename Ti, af_op_t op>
Array<To> *createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs)
{
    BinOp<To, Ti, op> bop;

    JIT::Node *lhs_node = lhs.getNode();
    JIT::Node *rhs_node = rhs.getNode();
    JIT::BinaryNode *node = new JIT::BinaryNode(dtype_traits<To>::getName(),
                                                bop.name(),
                                                lhs_node,
                                                rhs_node);

    return createNodeArray<To>(lhs.dims(), reinterpret_cast<JIT::Node *>(node));
}

}
