#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <bitset>

// 初始化随机数生成器
// inline static std::mt19937 gen(std::random_device{}());
inline static std::mt19937 gen(0);

// 均匀分布整数
template <int min, int max>
inline static int uniform_int_distribution()
{
    static std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// 正态分布
template <auto mean, auto stddev>
inline static double normal_distribution()
{
    static std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
}

inline static constexpr double divSqrt2 = 0.7071067811865475;

// ------------------- concept -------------------

template<typename T>
concept IntegralValue = std::integral<std::remove_reference_t<T>>;

template<typename T>
concept FirstElementIsIntegral = requires(T a) {
    { a[0] } -> IntegralValue;
};

// ------------------- tagExtractor -------------------

template <typename Tag, typename... Args>
struct tagExtractor;

// 值未能匹配，最终返回默认值
template <typename T, template <T> class Tag, T Value>
struct tagExtractor<Tag<Value>>
{
    static constexpr T value = Value;
};

// 值匹配成功
template <typename T, template <T> class Tag, T Value, T Value2,
          typename... Args>
struct tagExtractor<Tag<Value>, Tag<Value2>, Args...>
{
    static constexpr T value = Value2;
};

// 类型未能匹配，最终返回默认值
template <template <typename> class Tag, typename T>
struct tagExtractor<Tag<T>>
{
    using type = T;
};

// 类型未能匹配，最终返回默认值，多个参数版本
// 注意多个参数版本非常特殊地会将Tag<>保留
template <template <typename...> class Tag, typename... Args>
struct tagExtractor<Tag<Args...>>
{
    using type = Tag<Args...>;
};

// 类型匹配成功，单个参数版本
template <template <typename> class Tag, typename T, typename T2,
          typename... Args>
struct tagExtractor<Tag<T>, Tag<T2>, Args...>
{
    using type = T2;
};

// 类型匹配成功，多个参数版本
template <template <typename...> class Tag, typename... Args, typename... Args2,
          typename... Args3>
struct tagExtractor<Tag<Args...>, Tag<Args2...>, Args3...>
{
    using type = Tag<Args2...>;
};

// 匹配失败，类型不符，继续递归
template <typename Tag, typename Tag2, typename... Args>
struct tagExtractor<Tag, Tag2, Args...> : tagExtractor<Tag, Args...>
{};

// ------------------- Detection -------------------

// specify using float or double
template <typename PrecType>
struct Prec;

// specify using real domain or complex domain
struct RD;
struct CD;

template <typename DomainType>
struct Dom
{
    using type = DomainType;
};

template <size_t N>
struct Rx
{
    static constexpr size_t value = N;
};

template <size_t N>
struct Tx
{
    static constexpr size_t value = N;
};

template< typename Prec>
struct QAM16
{
    inline static constexpr size_t bitLength = 4;
    inline static constexpr std::array<Prec, 4> symbolsRD = {
        -0.31622776601683794, -0.9486832980505138, 0.31622776601683794,
        0.9486832980505138};
};

template< typename Prec>
struct QAM64
{
    inline static constexpr size_t bitLength = 6;
    inline static constexpr std::array<Prec, 8> symbolsRD = {
        -0.4629100498862757, -0.1543033499620919, -0.7715167498104595,
        -1.0801234497346432, 0.1543033499620919, 0.4629100498862757,
        0.7715167498104595, 1.0801234497346432};
};

template< typename Prec>
struct QAM256
{
    inline static constexpr size_t bitLength = 8;
    inline static constexpr std::array<Prec, 16> symbolsRD = {
        -0.3834824944236852, -0.5368754921931592, -0.2300894966542111,
        -0.07669649888473704, -0.8436614877321074, -0.6902684899626333,
        -0.9970544855015815, -1.1504474832710556, 0.3834824944236852,
        0.5368754921931592, 0.2300894966542111, 0.07669649888473704,
        0.8436614877321074, 0.6902684899626333, 0.9970544855015815,
        1.1504474832710556};
};

template <typename ModType>
struct Mod
{
    using type = ModType;
};

template <typename... Args>
class Detection_s;

template <typename PrecInput, size_t RxAntNumInput, size_t TxAntNumInput,
          typename ModTypeInput>
class Detection_s<Prec<PrecInput>, Dom<RD>, Rx<RxAntNumInput>,
                  Tx<TxAntNumInput>, Mod<ModTypeInput>>
{
public:
    inline static constexpr size_t RxAntNum = RxAntNumInput;
    inline static constexpr size_t TxAntNum = TxAntNumInput;
    using ModType = ModTypeInput;
    using PrecType = PrecInput;

    inline static constexpr auto symbolsRD = ModType::symbolsRD;

    Eigen::Vector<size_t, 2 * TxAntNum> TxIndices;
    Eigen::Vector<PrecType, 2 * TxAntNum> TxSymbols;
    Eigen::Vector<PrecType, 2 * RxAntNum> RxSymbols;
    Eigen::Matrix<PrecType, 2 * RxAntNum, 2 * TxAntNum> H;

    double Nv = 1;
    double sqrtNvDiv2 = std::sqrt(Nv / 2);

    void setSNR(double SNRdB)
    {
        Nv = TxAntNum * RxAntNum /
             (std::pow(10, SNRdB / 10) * ModType::bitLength * TxAntNum);
        sqrtNvDiv2 = std::sqrt(Nv / 2);
    }

    inline void generateTx()
    {
        std::generate(TxIndices.begin(), TxIndices.end(), []() {
            return uniform_int_distribution<0, ModType::symbolsRD.size() - 1>();
        });
        std::transform(TxIndices.begin(), TxIndices.end(), TxSymbols.begin(),
                       [](size_t index) { return symbolsRD[index]; });
    }

    inline void generateH()
    {

        for (size_t j = 0; j < TxAntNum; j++)
        {
            for (size_t i = 0; i < RxAntNum; i++)
            {
                auto temp = normal_distribution<0, divSqrt2>();

                H(i, j) = temp;
                H(i + RxAntNum, j + TxAntNum) = temp;

                temp = normal_distribution<0, divSqrt2>();
                H(i, j + TxAntNum) = temp;
                H(i + RxAntNum, j) = -temp;
            }
        }
    }

    inline void generateRx()
    {
        RxSymbols = H * TxSymbols;
        RxSymbols += Eigen::Vector<PrecType, 2 * RxAntNum>::NullaryExpr([&](size_t i) { return normal_distribution<0, 1>() * sqrtNvDiv2; });
    }

    inline void generate()
    {
        generateTx();
        generateH();
        generateRx();
    }

    template <typename T>
        requires  (!FirstElementIsIntegral<T>)
    inline auto judge(T &symbolsEst)
    {

        static std::array<size_t, 2 * TxAntNum> wrongBits;

        std::transform(
            symbolsEst.begin(), symbolsEst.end(), TxIndices.begin(),
            wrongBits.begin(), [](auto symbol, size_t index) {
                auto closest = std::min_element(symbolsRD.begin(), symbolsRD.end(),
                                                [symbol](auto x, auto y) {
                                                    return std::abs(x - symbol) <
                                                           std::abs(y - symbol);
                                                }) -
                               symbolsRD.begin();
                return std::bitset<ModType::bitLength>(closest ^ index).count();
            });

        return std::accumulate(wrongBits.begin(), wrongBits.end(), 0);
    }

    template <typename T>
        requires  (FirstElementIsIntegral<T>)
    inline auto judge(T &bitsEst)
    {
        static std::array<size_t, 2 * TxAntNum> wrongBits;

        std::transform(bitsEst.begin(), bitsEst.end(), TxIndices.begin(),
                       wrongBits.begin(), [](auto bits, size_t index) {
                           return std::bitset<ModType::bitLength>(bits ^ index).count();
                       });

        return std::accumulate(wrongBits.begin(), wrongBits.end(), 0);
    }
};

template <typename... Args>
struct DetectionInputHelper
{
    inline static constexpr auto RxAntNum = tagExtractor<Rx<4>, Args...>::value;
    inline static constexpr auto TxAntNum = tagExtractor<Tx<4>, Args...>::value;

    using DomainType = tagExtractor<Dom<RD>, Args...>::type;
    using PrecType = tagExtractor<Prec<float>, Args...>::type;
    using ModType = tagExtractor<Mod<QAM16<PrecType>>, Args...>::type;
    static_assert(RxAntNum > 0, "RxAntNum must be greater than 0");
    static_assert(TxAntNum > 0, "TxAntNum must be greater than 0");
    static_assert(RxAntNum >= TxAntNum,
                  "RxAntNum must be greater than or equal to TxAntNum");

    static_assert(std::is_same<ModType, QAM16<PrecType>>::value ||
                      std::is_same<ModType, QAM64<PrecType>>::value ||
                      std::is_same<ModType, QAM256<PrecType>>::value,
                  "ModType must be QAM16, QAM64 or QAM256");
    static_assert(std::is_same<DomainType, RD>::value ||
                      std::is_same<DomainType, CD>::value,
                  "DomainType must be RD or CD");
    static_assert(std::is_same<PrecType, float>::value ||
                      std::is_same<PrecType, double>::value,
                  "PrecType must be float or double");

    using type = Detection_s<Prec<PrecType>, Dom<DomainType>, Rx<RxAntNum>,
                             Tx<TxAntNum>, Mod<ModType>>;
};

template <typename... Args>
using Detection = typename DetectionInputHelper<Args...>::type;