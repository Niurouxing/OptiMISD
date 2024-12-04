#include "OptiMISD.h"
#include <iostream>
#include <cstdlib>
#include <chrono>

static constexpr size_t TxAntNum = 128;
static constexpr size_t RxAntNum = 1024;

using QAM = QAM256<float>;

int main(int argc, char* argv[])
{
    // 默认参数设置
    int sample = 1000;        
    int snr = 20;             
    int playout = 1000;       
    float c = 1.41f;          
    float r = 1.0f;           
    unsigned int seed = 42;   

    // 使用命令行参数替代默认值（如果提供）
    if (argc > 1) sample = atoi(argv[1]);
    if (argc > 2) snr = atoi(argv[2]);
    if (argc > 3) playout = atoi(argv[3]);
    if (argc > 4) c = atof(argv[4]);
    if (argc > 5) r = atof(argv[5]);
    if (argc > 6) seed = atoi(argv[6]);

    set_random_seed(seed);

    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
    det.setSNR(snr);

    auto mcts = OptiMISD<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>(playout, c, r);


    int err_frames = 0;
    int err_bits = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < sample; i++)
    {
        det.generate();
        auto list = mcts.execute(det);
        auto err = det.judge(list);

        err_frames += err > 0;
        err_bits += err;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Total Err Bits: " << err_bits << std::endl;
    std::cout << "Total Err Frames: " << err_frames << std::endl;
    std::cout << "BER: " << static_cast<float>(err_bits) / (sample * TxAntNum * QAM::bitLength) << std::endl;
    std::cout << "FER: " << static_cast<float>(err_frames) / sample << std::endl;
    
    return 0;
}
