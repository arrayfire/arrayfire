#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>
#include <TNJ/BinaryNode.hpp>

namespace cpu
{

    template<typename To, typename Ti>
    struct BinOp<To, Ti, af_cplx2_t>
    {
        To eval(Ti lhs, Ti rhs)
        {
            return To(lhs, rhs);
        }
    };

    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        TNJ::Node *lhs_node = lhs.getNode();
        TNJ::Node *rhs_node = rhs.getNode();

        TNJ::BinaryNode<To, Ti, af_cplx2_t> *node =
            new TNJ::BinaryNode<To, Ti, af_cplx2_t>(lhs_node, rhs_node);

        return createNodeArray<To>(lhs.dims(), reinterpret_cast<TNJ::Node *>(node));
    }
}
