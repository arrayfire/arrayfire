/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "BufferNode.hpp"
#include "Node.hpp"
#include <iomanip>
#include <mutex>

namespace opencl
{

namespace JIT
{
    class ShiftNode : public Node
    {
    private:

        Node_ptr m_buffer_node;
        const std::array<int, 4> m_shifts;

    public:

        ShiftNode(const char *type_str,
                  const char *name_str,
                  Node_ptr buffer_node,
                  const std::array<int, 4> shifts)
            : Node(type_str, name_str, 0, {}),
              m_buffer_node(buffer_node),
              m_shifts(shifts)
        {
        }

        bool isBuffer() { return false; }

        void setData(KParam info, std::shared_ptr<cl::Buffer> data, const unsigned bytes, bool is_linear)
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode *>(node_ptr)->setData(info, data, bytes, is_linear);
        }

        bool isLinear(dim_t dims[4])
        {
            return false;
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream, int id, bool is_linear)
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode *>(node_ptr)->genParams(kerStream, id, is_linear);
            for (int i = 0; i < 4; i++) {
                kerStream << "int shift" << id << "_" << i << ",\n";
            }
        }

        int setArgs(cl::Kernel &ker, int id, bool is_linear)
        {
            auto node_ptr = m_buffer_node.get();
            int curr_id = dynamic_cast<BufferNode *>(node_ptr)->setArgs(ker, id, is_linear);
            for (int i = 0; i < 4; i++) {
                ker.setArg(curr_id + i, m_shifts[i]);
            }
            return curr_id + 4;
        }

        void genOffsets(std::stringstream &kerStream, int id, bool is_linear)
        {
            std::string idx_str = std::string("idx") + std::to_string(id);
            std::string info_str = std::string("iInfo") + std::to_string(id);
            std::string id_str = std::string("sh_id_") + std::to_string(id) + "_";
            std::string shift_str = std::string("shift") + std::to_string(id) + "_";

            for (int i = 0; i < 4; i++) {
                kerStream << "int " << id_str << i
                          << " = __circular_mod(id" << i
                          << " + " << shift_str << i
                          << ", " << info_str << ".dims[" << i << "]"
                          << ");\n";
            }

            kerStream << "int " << idx_str << " = "
                      << "(" << id_str << "3 < " << info_str << ".dims[3]) * "
                      << info_str << ".strides[3] * " << id_str << "3;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "2 < " << info_str << ".dims[2]) * "
                      << info_str << ".strides[2] * " << id_str << "2;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "1 < " << info_str << ".dims[1]) * "
                      << info_str << ".strides[1] * " << id_str << "1;\n";
            kerStream << idx_str  << " += "
                      << "(" << id_str << "0 < " << info_str << ".dims[0]) * "
                      << id_str << "0 + " << info_str << ".offset;"
                      << "\n";
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << m_type_str << " val" << ids.id << " = "
                      << "in" << ids.id << "[idx" << ids.id << "];"
                      << "\n";
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            auto node_ptr = m_buffer_node.get();
            dynamic_cast<BufferNode *>(node_ptr)->getInfo(len, buf_count, bytes);
        }
    };
}

}
