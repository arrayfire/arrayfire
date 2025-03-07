/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <memory.hpp>
#include <map>
#include <set>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
class LabelNode {
   private:
    T label;
    T minLabel;
    unsigned rank;
    LabelNode* parent;

   public:
    LabelNode() : label(0), minLabel(0), rank(0), parent(this) {}
    LabelNode(T label) : label(label), minLabel(label), rank(0), parent(this) {}

    T getLabel() { return label; }

    T getMinLabel() { return minLabel; }

    LabelNode* getParent() { return parent; }

    unsigned getRank() { return rank; }

    void setMinLabel(T l) { minLabel = l; }

    void setParent(LabelNode* p) { parent = p; }

    void setRank(unsigned r) { rank = r; }
};

template<typename T>
static LabelNode<T>* find(LabelNode<T>* x) {
    if (x->getParent() != x) x->setParent(find(x->getParent()));
    return x->getParent();
}

template<typename T>
static void setUnion(LabelNode<T>* x, LabelNode<T>* y) {
    LabelNode<T>* xRoot = find(x);
    LabelNode<T>* yRoot = find(y);
    if (xRoot == yRoot) return;

    T xMinLabel = xRoot->getMinLabel();
    T yMinLabel = yRoot->getMinLabel();
    xRoot->setMinLabel(std::min(xMinLabel, yMinLabel));
    yRoot->setMinLabel(std::min(xMinLabel, yMinLabel));

    if (xRoot->getRank() < yRoot->getRank())
        xRoot->setParent(yRoot);
    else if (xRoot->getRank() > yRoot->getRank())
        yRoot->setParent(xRoot);
    else {
        yRoot->setParent(xRoot);
        xRoot->setRank(xRoot->getRank() + 1);
    }
}

template<typename T>
void regions(Param<T> out, CParam<char> in, af_connectivity connectivity) {
    const af::dim4 inDims = in.dims();
    const char* inPtr     = in.get();
    T* outPtr             = out.get();

    // Map labels
    typedef typename std::unique_ptr<LabelNode<T>> UnqLabelPtr;
    typedef typename std::map<T, UnqLabelPtr> LabelMap;
    typedef typename LabelMap::iterator LabelMapIterator;

    LabelMap lmap;

    // Initial label
    T label = (T)1;

    for (int j = 0; j < (int)inDims[1]; j++) {
        for (int i = 0; i < (int)inDims[0]; i++) {
            int idx = j * inDims[0] + i;
            if (inPtr[idx] != 0) {
                std::vector<T> l;

                // Test neighbors
                if (i > 0 && outPtr[j * (int)inDims[0] + i - 1] > 0)
                    l.push_back(outPtr[j * inDims[0] + i - 1]);
                if (j > 0 && outPtr[(j - 1) * (int)inDims[0] + i] > 0)
                    l.push_back(outPtr[(j - 1) * inDims[0] + i]);
                if (connectivity == AF_CONNECTIVITY_8 && i > 0 && j > 0 &&
                    outPtr[(j - 1) * inDims[0] + i - 1] > 0)
                    l.push_back(outPtr[(j - 1) * inDims[0] + i - 1]);
                if (connectivity == AF_CONNECTIVITY_8 &&
                    i < (int)inDims[0] - 1 && j > 0 &&
                    outPtr[(j - 1) * inDims[0] + i + 1] != 0)
                    l.push_back(outPtr[(j - 1) * inDims[0] + i + 1]);

                if (!l.empty()) {
                    T minl = l[0];
                    for (size_t k = 0; k < l.size(); k++) {
                        minl                        = std::min(l[k], minl);
                        LabelMapIterator currentMap = lmap.find(l[k]);
                        LabelNode<T>* node          = currentMap->second.get();
                        // Group labels of the same region under a disjoint set
                        for (size_t m = k + 1; m < l.size(); m++)
                            setUnion(node, lmap.find(l[m])->second.get());
                    }
                    // Set label to smallest neighbor label
                    outPtr[idx] = minl;
                } else {
                    // Insert new label in map
                    lmap.insert(std::make_pair(
                        label, UnqLabelPtr(new LabelNode<T>(label))));
                    outPtr[idx] = label++;
                }
            }
        }
    }

    std::set<T> removed;

    for (int j = 0; j < (int)inDims[1]; j++) {
        for (int i = 0; i < (int)inDims[0]; i++) {
            int idx = j * (int)inDims[0] + i;
            if (inPtr[idx] != 0) {
                T l                         = outPtr[idx];
                LabelMapIterator currentMap = lmap.find(l);

                if (currentMap != lmap.end()) {
                    LabelNode<T>* node = currentMap->second.get();

                    LabelNode<T>* nodeRoot = find(node);
                    outPtr[idx]            = nodeRoot->getMinLabel();

                    // Mark removed labels (those that are part of a region
                    // that contains a smaller label)
                    if (node->getMinLabel() < l || nodeRoot->getMinLabel() < l)
                        removed.insert(l);
                    if (node->getLabel() > node->getMinLabel())
                        removed.insert(node->getLabel());
                }
            }
        }
    }

    // Calculate final neighbors (ensure final labels are sequential)
    for (int j = 0; j < (int)inDims[1]; j++) {
        for (int i = 0; i < (int)inDims[0]; i++) {
            int idx = j * (int)inDims[0] + i;
            if (outPtr[idx] > 0) {
                outPtr[idx] -=
                    distance(removed.begin(), removed.lower_bound(outPtr[idx]));
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
