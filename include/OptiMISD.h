#pragma once

#include "Eigen/Core"
#include "utils.h"
#include <cmath>

// ------------------- MCTS_node -------------------

template <typename QAM, typename PrecInput>
struct Node
{
    PrecInput PED;                                        // 节点的代价值
    PrecInput score;                                      // 节点的分数
    std::array<int, QAM::symbolsRD.size()> child_indices; // 子节点的索引，-1 表示未初始化

    int saved_child_num; // 已经保存的子节点数量
    int visited_times;   // 访问次数
    bool full_expanded;  // 是否已经扩展完毕

    PrecInput node_data; // 存储于节点的符号

    // 默认构造函数
    Node()
        : PED(0), score(-1e9), saved_child_num(0), visited_times(0), full_expanded(false), node_data(0)
    {
        child_indices.fill(-1);
    }

    // 新的构造函数
    Node(PrecInput PED, PrecInput node_data, int visited_times = 0, int saved_child_num = 0, int child_index = -1, bool full_expanded = false, PrecInput score = -1e9)
        : PED(PED), node_data(node_data), visited_times(visited_times), saved_child_num(saved_child_num), full_expanded(full_expanded), score(score)
    {
        child_indices.fill(-1);
        if (saved_child_num > 0 && child_index != -1)
        {
            child_indices[0] = child_index;
        }
    }
};

// ------------------- OptiMISD -------------------

template <typename... Args>
class OptiMISD_s;

template <typename PrecInput, size_t RxAntNum, size_t TxAntNum, typename ModType>
class OptiMISD_s<Prec<PrecInput>, Rx<RxAntNum>, Tx<TxAntNum>, Mod<ModType>>
{
public:
    static inline constexpr size_t ConSize = ModType::symbolsRD.size();

    // QR decomposition
    Eigen::Matrix<PrecInput, 2 * RxAntNum, 2 * RxAntNum> bQ;
    Eigen::Matrix<PrecInput, 2 * RxAntNum, 2 * TxAntNum> bR;

    Eigen::Matrix<PrecInput, 2 * RxAntNum, 2 * TxAntNum> Q;
    Eigen::Matrix<PrecInput, 2 * TxAntNum, 2 * TxAntNum> R;

    Eigen::Vector<PrecInput, 2 * TxAntNum> z;

    static constexpr int game_length = 2 * TxAntNum;

    std::vector<Node<ModType, PrecInput>> nodes;

    int max_playout;
    PrecInput c;
    PrecInput r;

    constexpr static int expand = 1;

    std::array<int, game_length> accumulated_node_indices;
    std::array<PrecInput, game_length> accumulated_node_data;

    OptiMISD_s(int max_playout, PrecInput c, PrecInput r)
    {
        this->max_playout = max_playout;
        this->c = c;
        this->r = r;

        nodes.reserve(100000);

        // 创建一个空的根节点
        nodes.emplace_back();
    }

    auto execute(const auto &det)
    {
        nodes.clear();
        nodes.emplace_back();

        const auto &H = det.H;

        // QR decomposition
        Eigen::HouseholderQR<Eigen::Matrix<PrecInput, 2 * RxAntNum, 2 * TxAntNum>> qr(H);
        bQ = qr.householderQ();
        bR = qr.matrixQR().template triangularView<Eigen::Upper>();

        // slice and multiply
        Q = bQ.leftCols(2 * TxAntNum);
        R = bR.topRows(2 * TxAntNum);

        // z = Q' * RxSymbol;
        z = (Q.transpose() * det.RxSymbols);

        bool play_exit = false;
        PrecInput max_PED_allowed = r * det.Nv * 2 * det.TxAntNum;

        for (int playout = 0; playout < max_playout; playout++)
        {
            auto *current_node = &nodes[0];

            int step = 0;
            for (step = 0; step < game_length; step++)
            {

                if (current_node->full_expanded == false || step == game_length - 1)
                {
                    // unexpanded or the last step
                    break;
                }
                else
                {
                    int candidates_num = current_node->saved_child_num;

                    if (candidates_num == 1)
                    {
                        accumulated_node_indices[step] = current_node->child_indices[0];
                        accumulated_node_data[step] = nodes[current_node->child_indices[0]].node_data;
                        current_node = &nodes[current_node->child_indices[0]]; // 更新 current_node
                        current_node->visited_times++;
                    }
                    else
                    {
                        // 选择最大的UCB值
                        PrecInput max_ucb = -1;
                        int max_ucb_index = -1;

                        for (int i = 0; i < candidates_num; i++)
                        {

                            auto &child_node = nodes[current_node->child_indices[i]];

                            PrecInput ucb = child_node.score + c * std::sqrt(std::log(1 + current_node->visited_times)) / (std::powf(2, std::floor(std::log2(1 + child_node.visited_times))) ); 
\

                            if (ucb > max_ucb)
                            {
                                max_ucb = ucb;
                                max_ucb_index = current_node->child_indices[i];
                            }
                        }

                        accumulated_node_indices[step] = max_ucb_index;
                        accumulated_node_data[step] = nodes[max_ucb_index].node_data;
                        current_node = &nodes[max_ucb_index];
                        current_node->visited_times++;
                    }
                }
            }

            PrecInput score_backuped;
            if (current_node->full_expanded == true)
            {
                score_backuped = current_node->score;
            }
            else
            {
                PrecInput PED_left = max_PED_allowed - current_node->PED;

                // calculate the shared part of R[2*TxAntNum-1-step][2*TxAntNum-1-step+1:2*TxAntNum-1] with nodeData[accumulatedNode[step:2*TxAntNum-1]]
                PrecInput shared_part = z(2 * TxAntNum - 1 - step);
                for (int i = 0; i < step; i++)
                {
                    shared_part -= accumulated_node_data[i] * R(2 * TxAntNum - 1 - step, 2 * TxAntNum - 1 - i); // R is upper triangular and the index is reversed
                }

                // calculate the play with least PED of the current node, and the PED must not exceed the maxPEDAllowed
                int best_play = -1;
                PrecInput best_PED = 1e9;

                for (int i = 0; i < ConSize; i++)
                {
                    // check if the symbol is already as the child node of the current node
                    bool symbol_exist = false;
                    for (int j = 0; j < current_node->saved_child_num; j++)
                    {
                        if (ModType::symbolsRD[i] == nodes[current_node->child_indices[j]].node_data) // dangerous to compare float, but worthy
                        {
                            symbol_exist = true;
                            break;
                        }
                    }

                    if (symbol_exist != true)
                    {
                        PrecInput candidate_PED = std::abs(shared_part - ModType::symbolsRD[i] * R(2 * TxAntNum - 1 - step, 2 * TxAntNum - 1 - step));

                        if (candidate_PED < best_PED && candidate_PED < PED_left)
                        {
                            best_PED = candidate_PED;
                            best_play = i;
                        }
                    }
                }

                current_node->full_expanded = best_play == -1;

                if (best_play != -1)
                {
                    accumulated_node_data[step] = ModType::symbolsRD[best_play];

                    current_node->child_indices[current_node->saved_child_num] = nodes.size();
                    current_node->saved_child_num++;

                    nodes.emplace_back(current_node->PED + best_PED, accumulated_node_data[step], /*visited_times=*/1);

                    // 更新 current_node
                    current_node = &nodes.back();

                    accumulated_node_indices[step] = nodes.size() - 1;

                    // 下一步中，当前节点肯定会有一个子节点
                    current_node->saved_child_num = 1;
                    current_node->child_indices[0] = nodes.size();

                    // specialization for expand == 1
                    if constexpr (expand == 1)
                    {
                        // get the PED of the current node and add the PED of the best play
                        PrecInput PED = current_node->PED + best_PED;

                        // greedly select the child node with the least PED until the end of the game
                        for (int i = step + 1; i < game_length; i++)
                        {
                            PrecInput temp_double = z(2 * TxAntNum - 1 - i);
                            PrecInput current_best_PED = 1e9;

                            for (int j = 0; j < i; j++)
                            {
                                temp_double -= R(2 * TxAntNum - 1 - i, 2 * TxAntNum - 1 - j) * accumulated_node_data[j];
                            }

                            // find out the symbol with the least PED in layer i
                            for (int j = 0; j < ConSize; j++)
                            {
                                PrecInput candidate_PED = std::abs(temp_double - ModType::symbolsRD[j] * R(2 * TxAntNum - 1 - i, 2 * TxAntNum - 1 - i));

                                if (candidate_PED < current_best_PED)
                                {
                                    current_best_PED = candidate_PED;
                                    accumulated_node_data[i] = ModType::symbolsRD[j];
                                }
                            }

                            PED += current_best_PED;

                            // 创建新节点并添加到 nodes 中
                            int saved_child_num = (i != game_length - 1) ? 1 : 0;
                            int child_index = (saved_child_num == 1) ? nodes.size() + 1 : -1;

                            nodes.emplace_back(PED, accumulated_node_data[i], /*visited_times=*/1, saved_child_num, child_index);

                            accumulated_node_indices[i] = nodes.size() - 1;
                        }

                        score_backuped = 1 - PED / game_length;

                        // back propagation

                        // update the score for all the nodes in accumulated_node_indices
                        for (int i = 0; i < game_length; i++)
                        {
                            // if the score_backuped is larger than the current score, update the score
                            if (score_backuped > nodes[accumulated_node_indices[i]].score)
                            {
                                nodes[accumulated_node_indices[i]].score = score_backuped;
                            }
                        }

                        continue;
                    }
                    else
                    {
                        // not implemented, leave this for future work
                    }
                }
            }
        }

        // search the tree, find the child with the highest score, store the symbol in accumulatedNodeData

        std::array<PrecInput, game_length> res;

        auto *current_node = &nodes[0];
        for (int i = 0; i < game_length; i++)
        {
            if (current_node->saved_child_num == 1)
            {
                res[game_length - i - 1] = nodes[current_node->child_indices[0]].node_data;
                current_node = &nodes[current_node->child_indices[0]];
            }
            else
            {
                PrecInput max_score = -1e9;
                int max_score_index = -1;

                for (int j = 0; j < current_node->saved_child_num; j++)
                {
                    if (nodes[current_node->child_indices[j]].score > max_score)
                    {
                        max_score = nodes[current_node->child_indices[j]].score;
                        max_score_index = current_node->child_indices[j];
                    }
                }

                res[game_length - i - 1] = nodes[max_score_index].node_data; // need to reverse the order since the QR is upper triangular and the search is from the end to the beginning
                current_node = &nodes[max_score_index];
            }
        }

        return res;
    }
};

template <typename... Args>
struct OptiMISDInputHelper
{
    inline constexpr static size_t RxAntNum = tagExtractor<Rx<4>, Args...>::value;
    inline constexpr static size_t TxAntNum = tagExtractor<Tx<4>, Args...>::value;

    using PrecType = tagExtractor<Prec<float>, Args...>::type;
    using ModType = tagExtractor<Mod<QAM16<PrecType>>, Args...>::type;

    using type = OptiMISD_s<Prec<PrecType>, Rx<RxAntNum>, Tx<TxAntNum>, Mod<ModType>>;
};

template <typename... Args>
using OptiMISD = typename OptiMISDInputHelper<Args...>::type;
