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
#include <array>
#include <vector>
#include "Node.hpp"

namespace cpu {

template<typename To, typename Ti, af_op_t op>
struct BinOp;

namespace jit {

template<typename To, typename Ti, af_op_t op>
class BinaryNode : public TNode<compute_t<To>> {
   protected:
    BinOp<compute_t<To>, compute_t<Ti>, op> m_op;
    TNode<compute_t<Ti>> *m_lhs, *m_rhs;

   public:
    BinaryNode(common::Node_ptr lhs, common::Node_ptr rhs)
        : TNode<compute_t<To>>(compute_t<To>(0),
                               std::max(lhs->getHeight(), rhs->getHeight()) + 1,
                               {{lhs, rhs}})
        , m_lhs(reinterpret_cast<TNode<compute_t<Ti>> *>(lhs.get()))
        , m_rhs(reinterpret_cast<TNode<compute_t<Ti>> *>(rhs.get())) {}

    void calc(int x, int y, int z, int w, int lim) final {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
        m_op.eval(this->m_val, m_lhs->m_val, m_rhs->m_val, lim);
    }

    void calc(int idx, int lim) final {
        UNUSED(idx);
        m_op.eval(this->m_val, m_lhs->m_val, m_rhs->m_val, lim);
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        UNUSED(kerString);
        UNUSED(ids);
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size)>
                    setArg) const override {
        UNUSED(is_linear);
        UNUSED(setArg);
        return start_id++;
    }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        UNUSED(kerStream);
        UNUSED(ids);
    }
};

}  // namespace jit

}  // namespace cpu
