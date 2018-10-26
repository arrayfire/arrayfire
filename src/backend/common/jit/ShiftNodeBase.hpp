/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <jit/BufferNode.hpp>
#include <jit/kernel_generators.hpp>

#include <iomanip>
#include <backend.hpp>

#include <array>
#include <memory>
#include <sstream>
#include <string>

namespace common {

    template<typename BufferNode>
    class ShiftNodeBase : public Node
    {
    private:
        std::shared_ptr<BufferNode> m_buffer_node;
        const std::array<int, 4> m_shifts;

    public:
        ShiftNodeBase(const char *type_str,
                  const char *name_str,
                  std::shared_ptr<BufferNode> buffer_node,
                  const std::array<int, 4> shifts)
            : Node(type_str, name_str, 0, {}),
              m_buffer_node(buffer_node),
              m_shifts(shifts)
        {
        }

        bool isLinear(dim_t dims[4]) const final
        {
            return false;
        }

        void genKerName(std::stringstream &kerStream, const common::Node_ids& ids) const final
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream, int id, bool is_linear) const final
        {
            m_buffer_node->genParams(kerStream, id, is_linear);
            for (int i = 0; i < 4; i++) {
                kerStream << "int shift" << id << "_" << i << ",\n";
            }
        }

        int setArgs(int start_id, bool is_linear,
                    std::function<void(int id, const void* ptr, size_t arg_size)> setArg) const {
            int curr_id = m_buffer_node->setArgs(start_id, is_linear, setArg);
            for (int i = 0; i < 4; i++) {
                const int &d = m_shifts[i];
                setArg(curr_id+i, static_cast<const void*>(&d), sizeof(int));
            }
            return curr_id + 4;
        }

        void genOffsets(std::stringstream &kerStream, int id, bool is_linear) const final
        {
            detail::generateShiftNodeOffsets(kerStream, id, is_linear, m_type_str);
        }

        void genFuncs(std::stringstream &kerStream, const common::Node_ids& ids) const final
        {
            detail::generateShiftNodeRead(kerStream, ids.id, m_type_str);
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes) const final
        {
            m_buffer_node->getInfo(len, buf_count, bytes);
        }
    };
}
