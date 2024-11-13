
#include "OptiMISD.h"
#include <iostream>

    #include <chrono>

static constexpr size_t TxAntNum = 8;
static constexpr size_t RxAntNum = 8;

using QAM = QAM16<float>;

int main()
{

    int sample = 1000;

    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
    det.setSNR(18);

    auto mcts = OptiMISD<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>(4000,4, 3);


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
 

    // std::cout << "Error rate: " << (float)err_sum / (sample * TxAntNum * QAM::bitLength) << std::endl;

    std::cout << "BER: " << (float)err_bits / (sample * TxAntNum * QAM::bitLength) << std::endl;
    std::cout << "FER: " << (float)err_frames / sample << std::endl;


 
    

 
 

   
 
 
    
}