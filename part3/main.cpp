#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

int main(int argc, char *argv[])
{
    auto program_start = std::chrono::high_resolution_clock::now();

    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <query.json> <passages.json> <K> <eps>\n";
        return 1;
    }

    auto processing_start = std::chrono::high_resolution_clock::now();
    // Load and parse query JSON
    std::ifstream query_ifs(argv[1]);
    if (!query_ifs)
    {
        std::cerr << "Error opening query file: " << argv[1] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1)
    {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[2]);
    if (!passages_ifs)
    {
        std::cerr << "Error opening passages file: " << argv[2] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1)
    {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Convert JSON array to a dict mapping id -> element
    std::unordered_map<int, json> dict;
    for (auto &elem : passages_json)
    {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }

    // Parse K and eps
    int k = std::stoi(argv[3]);
    double eps = std::stof(argv[4]);

    try
    {
        // Extract the query embedding
        auto query_obj = query_json[0];
        const size_t D = query_obj["embedding"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d)
        {
            query[d] = query_obj["embedding"][d].get<double>();
        }

        // 1. Extract the passage embedding and store it in alglib::real_2d_array, store the idx of each embedding in alglib::integer_1d_array
        const size_t N_points = passages_json.size();

        alglib::real_2d_array allPoints;
        alglib::integer_1d_array tags;

        allPoints.setlength((alglib::ae_int_t)N_points, (alglib::ae_int_t)D);
        tags.setlength((alglib::ae_int_t)N_points);

        for (size_t i = 0; i < N_points; ++i)
        {
            const auto &item = passages_json[i];

            tags[(alglib::ae_int_t)i] = item["id"].get<int>();

            const auto &emb = item["embedding"];

            if (emb.size() != D)
            {
                throw std::runtime_error("Inconsistent embedding dimension at row " + std::to_string(i));
            }
            for (size_t d = 0; d < D; ++d)
            {
                allPoints((alglib::ae_int_t)i, (alglib::ae_int_t)d) = emb[d].get<double>();
            }
        }

        // 2. Build the KD-tree (alglib::kdtree) from the passages embeddings using alglib::buildkdtree
        alglib::kdtree tree;
        alglib::kdtreebuildtagged(allPoints, tags, (int)N_points, (int)D, 0, 2, tree);

        // 3. Perform the k-NN search using alglib::knnsearch
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
        std::cout << "Performed k-NN search using alglib::knntreequeryaknn. Count: " << count << std::endl;

        // 4. Query the results
        //     - Get the index of each found neighbour  using alglib::kdtreequeryresultstags
        //     - Get the distance between each found neighbour and the query embedding using alglib::kdtreequeryresultsdists
        alglib::integer_1d_array idx;
        idx.setlength(k);
        alglib::kdtreequeryresultstags(tree, idx);

        alglib::real_1d_array dist;
        dist.setlength(k);
        alglib::kdtreequeryresultsdistances(tree, dist);

        for (int i = 0; i < count; ++i)
        {
            std::cout << "Neighbor " << i + 1 << ": id = " << idx[i] << ", dist = " << dist[i] << std::endl;
        }
    }
    catch (alglib::ap_error &e)
    {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    return 0;
}