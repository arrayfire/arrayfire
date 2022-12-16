
#include <Array.hpp>
#include <binary.hpp>
#include <common/jit/BinaryNode.hpp>
#include <complex.hpp>
#include <types.hpp>

#include <memory>

using af::dim4;
using af::dtype_traits;
using detail::Array;
using detail::BinOp;
using detail::cdouble;
using detail::cfloat;
using detail::createNodeArray;

using std::make_shared;

namespace arrayfire {
namespace common {
#ifdef AF_CPU
template<typename To, typename Ti, af_op_t op>
Array<To> createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs,
                           const af::dim4 &odims) {
    common::Node_ptr lhs_node = lhs.getNode();
    common::Node_ptr rhs_node = rhs.getNode();

    auto node =
        make_shared<detail::jit::BinaryNode<To, Ti, op>>(lhs_node, rhs_node);

    return createNodeArray<To>(odims, move(node));
}

#else

template<typename To, typename Ti, af_op_t op>
Array<To> createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs,
                           const af::dim4 &odims) {
    auto createBinary = [](std::array<Node_ptr, 2> &operands) -> Node_ptr {
        BinOp<To, Ti, op> bop;
        return std::make_shared<BinaryNode>(
            static_cast<af::dtype>(dtype_traits<To>::af_type), bop.name(),
            operands[0], operands[1], op);
    };

    Node_ptr out =
        common::createNaryNode<Ti, 2>(odims, createBinary, {&lhs, &rhs});
    return createNodeArray<To>(odims, out);
}

#endif

#define INSTANTIATE(To, Ti, op)                      \
    template Array<To> createBinaryNode<To, Ti, op>( \
        const Array<Ti> &lhs, const Array<Ti> &rhs, const dim4 &odims)

INSTANTIATE(cfloat, float, af_cplx2_t);
INSTANTIATE(cdouble, double, af_cplx2_t);

#define INSTANTIATE_ARITH(op)                                \
    INSTANTIATE(float, float, op);                           \
    INSTANTIATE(cfloat, cfloat, op);                         \
    INSTANTIATE(double, double, op);                         \
    INSTANTIATE(cdouble, cdouble, op);                       \
    INSTANTIATE(unsigned, unsigned, op);                     \
    INSTANTIATE(short, short, op);                           \
    INSTANTIATE(unsigned short, unsigned short, op);         \
    INSTANTIATE(unsigned long long, unsigned long long, op); \
    INSTANTIATE(long long, long long, op);                   \
    INSTANTIATE(unsigned char, unsigned char, op);           \
    INSTANTIATE(char, char, op);                             \
    INSTANTIATE(common::half, common::half, op);             \
    INSTANTIATE(int, int, op)

INSTANTIATE_ARITH(af_add_t);
INSTANTIATE_ARITH(af_sub_t);
INSTANTIATE_ARITH(af_mul_t);
INSTANTIATE_ARITH(af_div_t);
INSTANTIATE_ARITH(af_min_t);
INSTANTIATE_ARITH(af_max_t);

#undef INSTANTIATE_ARITH

#define INSTANTIATE_ARITH_REAL(op)                           \
    INSTANTIATE(float, float, op);                           \
    INSTANTIATE(double, double, op);                         \
    INSTANTIATE(unsigned, unsigned, op);                     \
    INSTANTIATE(short, short, op);                           \
    INSTANTIATE(unsigned short, unsigned short, op);         \
    INSTANTIATE(unsigned long long, unsigned long long, op); \
    INSTANTIATE(long long, long long, op);                   \
    INSTANTIATE(unsigned char, unsigned char, op);           \
    INSTANTIATE(char, char, op);                             \
    INSTANTIATE(common::half, common::half, op);             \
    INSTANTIATE(int, int, op)

INSTANTIATE_ARITH_REAL(af_rem_t);
INSTANTIATE_ARITH_REAL(af_pow_t);
INSTANTIATE_ARITH_REAL(af_mod_t);

#define INSTANTIATE_FLOATOPS(op)     \
    INSTANTIATE(float, float, op);   \
    INSTANTIATE(double, double, op); \
    INSTANTIATE(common::half, common::half, op)

INSTANTIATE_FLOATOPS(af_hypot_t);
INSTANTIATE_FLOATOPS(af_atan2_t);

#define INSTANTIATE_BITOP(op)                                \
    INSTANTIATE(unsigned, unsigned, op);                     \
    INSTANTIATE(short, short, op);                           \
    INSTANTIATE(unsigned short, unsigned short, op);         \
    INSTANTIATE(unsigned long long, unsigned long long, op); \
    INSTANTIATE(long long, long long, op);                   \
    INSTANTIATE(unsigned char, unsigned char, op);           \
    INSTANTIATE(char, char, op);                             \
    INSTANTIATE(int, int, op)

INSTANTIATE_BITOP(af_bitshiftl_t);
INSTANTIATE_BITOP(af_bitshiftr_t);
INSTANTIATE_BITOP(af_bitor_t);
INSTANTIATE_BITOP(af_bitand_t);
INSTANTIATE_BITOP(af_bitxor_t);
#undef INSTANTIATE_BITOP

#define INSTANTIATE_LOGIC(op)                  \
    INSTANTIATE(char, float, op);              \
    INSTANTIATE(char, double, op);             \
    INSTANTIATE(char, cfloat, op);             \
    INSTANTIATE(char, cdouble, op);            \
    INSTANTIATE(char, common::half, op);       \
    INSTANTIATE(char, unsigned, op);           \
    INSTANTIATE(char, short, op);              \
    INSTANTIATE(char, unsigned short, op);     \
    INSTANTIATE(char, unsigned long long, op); \
    INSTANTIATE(char, long long, op);          \
    INSTANTIATE(char, unsigned char, op);      \
    INSTANTIATE(char, char, op);               \
    INSTANTIATE(char, int, op)

INSTANTIATE_LOGIC(af_and_t);
INSTANTIATE_LOGIC(af_or_t);
INSTANTIATE_LOGIC(af_eq_t);
INSTANTIATE_LOGIC(af_neq_t);
INSTANTIATE_LOGIC(af_lt_t);
INSTANTIATE_LOGIC(af_le_t);
INSTANTIATE_LOGIC(af_gt_t);
INSTANTIATE_LOGIC(af_ge_t);

#undef INSTANTIATE_LOGIC
#undef INSTANTIATE

}  // namespace common
}  // namespace arrayfire
