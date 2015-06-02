/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <regions.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <map>
#include <set>
#include <algorithm>

using af::dim4;

namespace cpu
{

template<typename T>
class LabelNode
{
private:
    T label;
    T minLabel;
    unsigned rank;
    LabelNode* parent;

public:
    LabelNode() : label(0), minLabel(0), rank(0), parent(this) { }
    LabelNode(T label) : label(label), minLabel(label), rank(0), parent(this) { }

    T getLabel()
    {
        return label;
    }

    T getMinLabel()
    {
        return minLabel;
    }

    LabelNode* getParent()
    {
        return parent;
    }

    unsigned getRank()
    {
        return rank;
    }

    void setMinLabel(T l)
    {
        minLabel = l;
    }

    void setParent(LabelNode* p)
    {
        parent = p;
    }

    void setRank(unsigned r)
    {
        rank = r;
    }
};

template<typename T>
static LabelNode<T>* find(LabelNode<T>* x)
{
    if (x->getParent() != x)
        x->setParent(find(x->getParent()));
    return x->getParent();
}

template<typename T>
static void setUnion(LabelNode<T>* x, LabelNode<T>* y)
{
    LabelNode<T>* xRoot = find(x);
    LabelNode<T>* yRoot = find(y);
    if (xRoot == yRoot)
        return;

    T xMinLabel = xRoot->getMinLabel();
    T yMinLabel = yRoot->getMinLabel();
    xRoot->setMinLabel(min(xMinLabel, yMinLabel));
    yRoot->setMinLabel(min(xMinLabel, yMinLabel));

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
Array<T> regions(const Array<char> &in, af_connectivity connectivity)
{
    const dim4 in_dims = in.dims();

    // Create output placeholder
    Array<T> out = createValueArray(in_dims, (T)0);

    const char *in_ptr  = in.get();
          T    *out_ptr = out.get();

    // Map labels
    typedef typename std::map<T, LabelNode<T>* > label_map_t;
    typedef typename label_map_t::iterator label_map_iterator_t;

    label_map_t lmap;

    // Initial label
    T label = (T)1;

    for (int j = 0; j < (int)in_dims[1]; j++) {
        for (int i = 0; i < (int)in_dims[0]; i++) {
            int idx = j * in_dims[0] + i;
            if (in_ptr[idx] != 0) {
                std::vector<T> l;

                // Test neighbors
                if (i > 0 && out_ptr[j * (int)in_dims[0] + i-1] > 0)
                    l.push_back(out_ptr[j * in_dims[0] + i-1]);
                if (j > 0 && out_ptr[(j-1) * (int)in_dims[0] + i] > 0)
                    l.push_back(out_ptr[(j-1) * in_dims[0] + i]);
                if (connectivity == AF_CONNECTIVITY_8 && i > 0 && j > 0 && out_ptr[(j-1) * in_dims[0] + i-1] > 0)
                    l.push_back(out_ptr[(j-1) * in_dims[0] + i-1]);
                if (connectivity == AF_CONNECTIVITY_8 && i < (int)in_dims[0] - 1 && j > 0 && out_ptr[(j-1) * in_dims[0] + i+1] != 0)
                    l.push_back(out_ptr[(j-1) * in_dims[0] + i+1]);

                if (!l.empty()) {
                    T minl = l[0];
                    for (size_t k = 0; k < l.size(); k++) {
                        minl = min(l[k], minl);
                        label_map_iterator_t cur_map = lmap.find(l[k]);
                        LabelNode<T> *node = cur_map->second;
                        // Group labels of the same region under a disjoint set
                        for (size_t m = k+1; m < l.size(); m++)
                            setUnion(node, lmap.find(l[m])->second);
                    }
                    // Set label to smallest neighbor label
                    out_ptr[idx] = minl;
                }
                else {
                    // Insert new label in map
                    LabelNode<T> *node = new LabelNode<T>(label);
                    lmap.insert(std::pair<T, LabelNode<T>* >(label, node));
                    out_ptr[idx] = label++;
                }
            }
        }
    }

    std::set<T> removed;

    for (int j = 0; j < (int)in_dims[1]; j++) {
        for (int i = 0; i < (int)in_dims[0]; i++) {
            int idx = j * (int)in_dims[0] + i;
            if (in_ptr[idx] != 0) {
                T l = out_ptr[idx];
                label_map_iterator_t cur_map = lmap.find(l);

                if (cur_map != lmap.end()) {
                    LabelNode<T>* node = cur_map->second;

                    LabelNode<T>* node_root = find(node);
                    out_ptr[idx] = node_root->getMinLabel();

                    // Mark removed labels (those that are part of a region
                    // that contains a smaller label)
                    if (node->getMinLabel() < l || node_root->getMinLabel() < l)
                        removed.insert(l);
                    if (node->getLabel() > node->getMinLabel())
                        removed.insert(node->getLabel());
                }
            }
        }
    }

    // Calculate final neighbors (ensure final labels are sequential)
    for (int j = 0; j < (int)in_dims[1]; j++) {
        for (int i = 0; i < (int)in_dims[0]; i++) {
            int idx = j * (int)in_dims[0] + i;
            if (out_ptr[idx] > 0) {
                out_ptr[idx] -= distance(removed.begin(), removed.lower_bound(out_ptr[idx]));
            }
        }
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> regions<T>(const Array<char> &in, af_connectivity connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(int   )
INSTANTIATE(uint  )

}
