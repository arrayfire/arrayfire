#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cuda.hpp>
#include <JIT/BinaryNode.hpp>

namespace cuda
{
    template<typename T> static const char *cplx2_name() { return "___noop"; }
    template<> const char *cplx2_name<cfloat>() { return "___cplxCss"; }
    template<> const char *cplx2_name<cdouble>() { return "___cplxZdd"; }

    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        JIT::Node *lhs_node = lhs.getNode();
        JIT::Node *rhs_node = rhs.getNode();

        JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                    cplx2_name<To>(),
                                                    lhs_node,
                                                    rhs_node, (int)(af_cplx2_t));

        return createNodeArray<To>(lhs.dims(), reinterpret_cast<JIT::Node *>(node));
    }
}
