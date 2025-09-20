
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>

template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};

// dynamic vector: runtime-D (global, set once at startup)
inline size_t &runtime_dim()
{
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }

    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};

// extract the “axis”-th coordinate or the scalar itself
template <typename T>
constexpr float getCoordinate(T const &e, size_t axis)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return e; // scalar case
    }
    else
    {
        return e[axis]; // vector case
    }
}

// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;

/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T> *buildKD(std::vector<std::pair<T, int>> &items, int depth = 0)
{
    if (items.empty())
        return nullptr;

    // set splittingAxis to be d % k for k-d data
    const size_t k = Embedding_T<T>::Dim();
    const size_t splittingAxis = depth % k;

    // compare function, comparing starting with splittingAxis
    auto cmp = [splittingAxis, k](const std::pair<T, int> &a, const std::pair<T, int> &b)
    {
        for (size_t off = 0; off < k; ++off)
        {
            const size_t dim = (splittingAxis + off) % k;
            const float va = getCoordinate(a.first, dim);
            const float vb = getCoordinate(b.first, dim);
            if (va < vb)
                return true;
            if (va > vb)
                return false;
        }
        return a.second < b.second;
    };

    // sort by splittingAxis in embedding
    std::sort(items.begin(), items.end(), cmp);

    // find median value using median index = size / 2
    std::size_t size = items.size();
    std::size_t mid = (size - 1) / 2;
    std::pair<T, int> median = items[mid];

    // build root node using median value
    Node<T> *root = new Node<T>();
    root->embedding = median.first;
    root->idx = median.second;

    // recursively build subtrees using left/right slices of vector
    std::vector<std::pair<T, int>> left(items.begin(), items.begin() + mid);
    std::vector<std::pair<T, int>> right(items.begin() + mid + 1, items.end());

    root->left = buildKD(left, depth + 1);
    root->right = buildKD(right, depth + 1);

    return root;
}

template <typename T>
void freeTree(Node<T> *node)
{
    if (!node)
        return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;

/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    if (!node)
        return;

    // get query
    const T &query = Node<T>::queryEmbedding;

    // set splittingAxis to be d % k for k-d data
    const size_t k = Embedding_T<T>::Dim();
    const size_t splittingAxis = depth % k;

    // process current node
    const float dist = Embedding_T<T>::distance(node->embedding, query);

    if ((int)heap.size() < K)
    {
        heap.push(PQItem{dist, node->idx});
    }
    else if (heap.top().first > dist)
    {
        heap.pop();
        heap.push(PQItem{dist, node->idx});
    }

    // recursively search the near subtree
    Node<T> *near;
    Node<T> *far;
    if (getCoordinate(query, splittingAxis) < getCoordinate(node->embedding, splittingAxis))
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

    // selectively decide whether to search far subtree
    const float planeDist = std::abs(getCoordinate(query, splittingAxis) - getCoordinate(node->embedding, splittingAxis));
    if ((int)heap.size() < K || planeDist < heap.top().first)
    {
        knnSearch(far, depth + 1, K, heap);
    }

    return;
}