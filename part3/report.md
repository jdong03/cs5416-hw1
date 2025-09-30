| Metric                    | Jason’s K-D Tree (ms) | ALGLIB (ms) | Difference (Jason − ALGLIB) (ms) |
| ------------------------- | --------------------: | ----------: | -------------------------------: |
| **Total elapsed**         |               5504.96 |     5874.61 |                      **−369.65** |
| **Process & parse input** |               5316.91 |     5509.22 |                      **−192.31** |
| **Build k-d tree**        |               174.293 |     345.213 |                     **−170.920** |
| **KNN search**            |               13.7568 |     19.8068 |                      **−6.0500** |

It seems like my K-D tree implementation was faster than the ALGLIB implementation in each individaul stage (process & parse input, build k-d tree, knn search), as well as the total time. The main gains come from a much faster tree build and KNN serach improvement since the parsing/processing input domiantes the overall time. A possible reason for this is the additional features built into the ALGLIB tree and functions which my leaner and simpler implementation may not offer.

|  k | ε (epsilon) | Search Time (ms) | Accuracy (% overlap with exact KNN) |
| -: | ----------: | ---------------: | ----------------------------------: |
|  1 |           0 |          20.6132 |                                100% |
|  1 |           1 |          20.5069 |                                100% |
|  1 |           5 |          9.47917 |                                100% |
|  5 |           0 |          20.4373 |                                100% |
|  5 |           1 |          20.7060 |                                100% |
|  5 |           5 |          14.1210 |                                100% |
| 10 |           0 |          21.1557 |                                100% |
| 10 |           1 |          20.8717 |                                100% |
| 10 |           5 |          13.7223 |                                100% |

All my runs with k ∈ {1, 5, 10} and ε ∈ {0, 1, 5} led to results with 100% accuracy. This is likely true due to the true neighbors being well separted, so relaxed pruning still found the exact same set. This reflects that increasing ε (at least minorly) does not neccesarily directly lead to decreased accuracy. It is also worth noting that increasing ε did not always lead to a faster search time either (k = 5 is a counter example). However, in general, it does seem like a larger ε lead to a decreased search time, especially jumping from 1 to 5. When tweaking this parameter, we must define the goal we are trying to accomplish clearly; for instance, if we require 100% accuracy and we need the k exact closest points we cannot use a higher ε even if it likely will provide 100% accuracy anyways. We can run some tests to try to pinpoint an ε value that balances our needs of speed versus accuracy in any given specific scenario.