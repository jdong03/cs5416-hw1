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
    if (!node)
        return;

    // get query
    const float query = Node::queryEmbedding;

    // recursively search the near subtree
    Node *near;
    Node *far;
    if (query < node->embedding)
    {
        near = node->left;
        far = node->right;
    }
    else
    {
        near = node->right;
        far = node->left;
    }

    knnSearch(near, depth + 1, K, heap);

    // process current node
    float dist = distance(node->embedding, query);

    if ((int)heap.size() < K)
    {
        heap.push(PQItem{dist, node->idx});
    }
    else if (heap.top().first > dist)
    {
        heap.pop();
        heap.push(PQItem{dist, node->idx});
    }

    // selectively decide whether to search far subtree
    float planeDist = distance(query, node->embedding);
    if ((int)heap.size() < K || planeDist < heap.top().first)
    {
        knnSearch(far, depth + 1, K, heap);
    }

    return;
}