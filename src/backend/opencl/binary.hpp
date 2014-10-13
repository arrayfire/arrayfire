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

#define BINARY_TYPE_1(fn)                      \
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


BINARY_TYPE_1(eq)
BINARY_TYPE_1(neq)
BINARY_TYPE_1(lt)
BINARY_TYPE_1(le)
BINARY_TYPE_1(gt)
BINARY_TYPE_1(ge)
BINARY_TYPE_1(add)
BINARY_TYPE_1(sub)
BINARY_TYPE_1(mul)
BINARY_TYPE_1(div)

#undef BINARY_TYPE_1

#define BINARY_TYPE_2(fn)                       \
    template<typename To, typename Ti>          \
    struct BinOp<To, Ti, af_##fn##_t>           \
    {                                           \
        const char *name()                      \
        {                                       \
            return "f"#fn;                      \
        }                                       \
    };                                          \
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


// FIXME: call #fn instead of "f"#fn for int types
BINARY_TYPE_2(min)
BINARY_TYPE_2(max)
BINARY_TYPE_2(pow)

template<typename Ti>
struct BinOp<cfloat, Ti, af_cplx2_t>
{
    const char *name()
    {
        return "__cplx2f";
    }
};

template<typename Ti>
struct BinOp<cdouble, Ti, af_cplx2_t>
{
    const char *name()
    {
        return "__cplx2";
    }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, af_cplx2_t>
{
    const char *name()
    {
        AF_ERROR("Invalid inputs to cplx2", AF_ERR_ARG);
        return "noop";
    }
};

template<typename To, typename Ti, af_op_t op>
Array<To> *createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs)
{
    BinOp<To, Ti, op> bop;

    JIT::Node *lhs_node = lhs.getNode();
    JIT::Node *rhs_node = rhs.getNode();
    JIT::BinaryNode *node = new JIT::BinaryNode(dtype_traits<To>::getName(),
                                                bop.name(),
                                                lhs_node,
                                                rhs_node, (int)(op));

    return createNodeArray<To>(lhs.dims(), reinterpret_cast<JIT::Node *>(node));
}

}
