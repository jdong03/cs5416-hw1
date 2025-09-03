#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;

float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}

constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e; // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node *buildKD(std::vector<std::pair<Embedding_T, int>> &items, int depth)
{
    if (items.empty())
        return nullptr;

    // set splittingAxis to 0 for 1-d data
    int splittingAxis = 0;

    // sort by embedding
    std::sort(items.begin(), items.end(),
              [splittingAxis](const std::pair<Embedding_T, int> &a, const std::pair<Embedding_T, int> &b)
              {
                  return a.first < b.first;
              });

    // find median value using median index = size / 2
    std::size_t size = items.size();
    std::size_t mid = size / 2;
    std::pair<Embedding_T, int> median = items[mid];

    // build root node using median value
    Node *root = new Node();
    root->embedding = median.first;
    root->idx = median.second;

    // recursively build subtrees using left/right slices of vector
    std::vector<std::pair<Embedding_T, int>> left(items.begin(), items.begin() + mid);
    std::vector<std::pair<Embedding_T, int>> right(items.begin() + mid + 1, items.end());

    root->left = buildKD(left, depth + 1);
    root->right = buildKD(right, depth + 1);

    return root;
}

void freeTree(Node *node)
{
    if (!node)
        return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
    return;
}