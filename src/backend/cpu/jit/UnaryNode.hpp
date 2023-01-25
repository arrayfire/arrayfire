/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include "Node.hpp"

#include <jit/BufferNode.hpp>
#include <vector>

namespace arrayfire {
namespace cpu {
template<typename To, typename Ti, af_op_t op>
struct UnOp {
    void eval(jit::array<compute_t<To>> &out,
              const jit::array<compute_t<Ti>> &in, int lim) const;
};

namespace jit {

template<typename To, typename Ti, af_op_t op>
class UnaryNode : public TNode<To> {
   protected:
    using arrayfire::common::Node::m_children;
    UnOp<To, Ti, op> m_op;

   public:
    UnaryNode(common::Node_ptr child)
        : TNode<To>(To(0), child->getHeight() + 1, {{child}}) {}

    std::unique_ptr<common::Node> clone() final {
        return std::make_unique<UnaryNode>(*this);
    }

    af_op_t getOp() const noexcept final { return op; }

    void calc(int x, int y, int z, int w, int lim) final {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
        auto child = static_cast<TNode<Ti> *>(m_children[0].get());
        m_op.eval(TNode<To>::m_val, child->m_val, lim);
    }

    void calc(int idx, int lim) final {
        UNUSED(idx);
        auto child = static_cast<TNode<Ti> *>(m_children[0].get());
        m_op.eval(TNode<To>::m_val, child->m_val, lim);
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        UNUSED(kerString);
        UNUSED(ids);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        UNUSED(kerStream);
        UNUSED(ids);
    }
};

}  // namespace jit
}  // namespace cpu
}  // namespace arrayfire
