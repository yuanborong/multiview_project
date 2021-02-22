#include <iostream>
#include <vector>
#include <vector>
#include <numeric>
#include <random>

template <typename VT>
void printv(std::vector<VT> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v.at(i) << " ";
    }
    std::cout << "\n";
}

// template <typename CV>
void convert_vec_to_float(std::vector<int> colsample_bytree_weight, int colsample_bytree_weight_factor, std::vector<float> &output)
{

    for (int i = 0; i < colsample_bytree_weight.size(); i++)
    {
        float fi = static_cast<float>(colsample_bytree_weight.at(i)) / static_cast<float>(colsample_bytree_weight_factor);
        output.push_back(fi);
    }
}

// template <typename N>
void normalize(std::vector<float> a, std::vector<float> &output)
{
    float sum_of_elems = 0;
    sum_of_elems = std::accumulate(a.begin(), a.end(),
                                   decltype(a)::value_type(0));
    for (int i = 0; i < a.size(); i++)
    {
        float item = a.at(i);
        float normalized = item / sum_of_elems;

        output.push_back(normalized);
    }
}

// template <typename CU>
void cumulative(std::vector<float> a, std::vector<float> &output)
{
    float running_total = 0;
    for (int i = 0; i < a.size(); i++)
    {
        running_total += a.at(i);
        output.push_back(running_total);
    }
}

// template <typename FI>
int find_index_less_or_equal(std::vector<float> a, float n)
{
    std::cout << "\n--func find_index_less_or_equal \n";
    std::cout << "\n n : " << n << "length of input:" << a.size() << "\n";
    if (n > 1)
    {
        std::cout << "\n ------------------------------------ n is bigger than cummularive sum; ERROR ";
        return a.size() - 1;
    }
    int i = 0;

    for (i; i < a.size(); i++)
    {
        float item = a.at(i);
        if (n < item)
        {
            std::cout << "n < item \n"
                      << n << "<" << item << '\n';
            break;
        }
    }
    std::cout << "\n n : " << n;
    std::cout << " ; i : " << i;
    return i;
}

template <typename TUU>
int choice_c(std::vector<TUU> input, std::vector<float> p)
{
    // std::cout << "\nChoice started\n";

    if (input.size() != p.size())
    {
        std::cout << "\ninput vector and probability vector size is not the same \n";
        return -1.0;
    }
    int conversion = 1; // will be used to divide by 1.0
    // std::vector<float> float_vector = {};
    std::vector<float> normalized = {};
    std::vector<float> cumulatived = {};

    // convert_vec_to_float(p, conversion, float_vector);
    normalize(p, normalized);
    // printv(normalized);

    cumulative(normalized, cumulatived);
    // printv(cumulatived);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1); //uniform distribution between 0 and 1
    float r = dis(gen);
    int index = find_index_less_or_equal(cumulatived, r);
    float selected;
    // selected = input.at(index);

    // std::cout << "\nChoice ended\n";
    return index;
}

// template <typename TUU>
// std::vector<TUU> choice(std::vector<TUU> a, int size, bool replace, std::vector<int> p)

template <typename TUU>
std::vector<TUU> choice_n(std::vector<TUU> input, int n, bool replace, std::vector<int> p)
{
    std::vector<float> pf = {};
    convert_vec_to_float(p, 1, pf);

    std::vector<TUU> output;
    for (int i = 0; i < n; i++)
    {

        int index = choice_c(input, pf);
        float item = input.at(index);
        output.push_back(item);
        input.erase(input.begin() + index);
        pf.erase(pf.begin() + index);
        //
    }
    return output;
}

int main()
{
    // std::vector<float> p = {0.00228131, 0.00456263, 0.00684394, 0.00912526, 0.0114066, 0.0136879, 0.0159692, 0.0182505, 0.0205318, 0.0228131, 0.0250945, 0.0273758, 0.0296571, 0.0319384, 0.0342197, 0.036501, 0.0387824, 0.0410637, 0.043345, 0.0456263, 0.0479076, 0.0501889, 0.0524703, 0.0547516, 0.0570329, 0.0593142, 0.0615955, 0.0638768, 0.0661581, 0.0684395, 0.0707208, 0.0730021, 0.0752834, 0.0775647, 0.079846, 0.0821274, 0.0836245, 0.0859058, 0.0874029, 0.0889, 0.0911813, 0.0934626, 0.0949598, 0.0972411, 0.0995224, 0.101804, 0.103301, 0.105582, 0.107863, 0.110145, 0.112426, 0.114707, 0.116989, 0.11927, 0.121551, 0.123833, 0.126114, 0.128395, 0.130677, 0.132958, 0.135239, 0.136736, 0.139018, 0.140515, 0.142796, 0.145077, 0.147359, 0.14964, 0.151921, 0.154203, 0.156484, 0.158765, 0.161047, 0.163328, 0.164825, 0.167106, 0.169388, 0.171669, 0.17395, 0.176231, 0.178513, 0.180794, 0.183075, 0.185357, 0.187638, 0.189919, 0.192201, 0.193698, 0.195979, 0.19826, 0.200542, 0.202823, 0.205104, 0.207386, 0.209667, 0.211948, 0.21423, 0.215727, 0.218008, 0.220289, 0.222571, 0.224068, 0.226349, 0.227846, 0.229343, 0.23084, 0.232337, 0.233835, 0.236116, 0.238397, 0.240679, 0.24296, 0.244457, 0.246738, 0.24902, 0.251301, 0.253582, 0.255864, 0.257361, 0.258858, 0.260355, 0.261852, 0.263349, 0.264846, 0.266343, 0.26784, 0.269338, 0.270835, 0.273116, 0.274613, 0.27611, 0.278392, 0.280673, 0.282954, 0.285235, 0.287517, 0.289798, 0.292079, 0.294361, 0.296642, 0.298923, 0.301205, 0.303486, 0.305767, 0.308049, 0.31033, 0.312611, 0.314893, 0.31639, 0.317887, 0.320168, 0.321665, 0.323946, 0.326228, 0.328509, 0.33079, 0.333072, 0.335353, 0.337634, 0.339131, 0.341413, 0.343694, 0.345975, 0.347472, 0.349754, 0.351251, 0.352748, 0.354245, 0.355742, 0.357239, 0.359521, 0.361018, 0.363299, 0.36558, 0.367862, 0.369359, 0.370856, 0.372353, 0.37385, 0.375347, 0.376844, 0.379126, 0.381407, 0.383688, 0.38597, 0.387467, 0.389748, 0.392029, 0.394311, 0.396592, 0.398089, 0.40037, 0.402652, 0.404149, 0.405646, 0.407143, 0.40864, 0.410137, 0.411635, 0.413132, 0.414629, 0.416126, 0.417623, 0.41912, 0.420617, 0.422114, 0.424396, 0.425893, 0.42739, 0.428887, 0.430384, 0.431881, 0.433378, 0.434875, 0.436373, 0.43787, 0.439367, 0.440864, 0.442361, 0.443858, 0.44614, 0.448421, 0.450702, 0.452983, 0.455265, 0.456762, 0.459043, 0.461324, 0.462822, 0.464319, 0.465816, 0.468097, 0.470378, 0.47266, 0.474941, 0.477222, 0.479504, 0.481785, 0.483282, 0.485563, 0.487845, 0.490126, 0.492407, 0.494689, 0.49697, 0.499251, 0.501533, 0.503814, 0.505311, 0.507592, 0.509874, 0.512155, 0.514436, 0.516718, 0.518215, 0.520496, 0.522777, 0.524274, 0.525771, 0.527269, 0.528766, 0.530263, 0.53176, 0.533257, 0.534754, 0.537035, 0.539317, 0.541598, 0.543879, 0.546161, 0.547658, 0.549939, 0.55222, 0.554502, 0.555999, 0.55828, 0.560561, 0.562843, 0.56434, 0.566621, 0.568902, 0.571184, 0.572681, 0.574178, 0.576459, 0.577956, 0.579453, 0.58095, 0.583232, 0.585513, 0.58701, 0.588507, 0.590004, 0.591501, 0.593783, 0.59528, 0.596777, 0.598274, 0.599771, 0.602052, 0.603549, 0.605046, 0.607328, 0.609609, 0.61189, 0.613387, 0.614885, 0.616382, 0.618663, 0.62016, 0.622441, 0.623938, 0.625436, 0.627717, 0.629214, 0.630711, 0.632992, 0.635274, 0.637555, 0.639052, 0.641333, 0.64283, 0.645112, 0.647393, 0.649674, 0.651956, 0.653453, 0.65495, 0.656447, 0.658728, 0.660225, 0.661722, 0.66322, 0.664717, 0.666214, 0.667711, 0.669208, 0.670705, 0.672202, 0.673699, 0.67598, 0.677478, 0.678975, 0.680472, 0.681969, 0.683466, 0.684963, 0.68646, 0.687957, 0.690238, 0.69252, 0.694017, 0.695514, 0.697011, 0.698508, 0.700005, 0.701502, 0.702999, 0.704497, 0.705994, 0.707491, 0.708988, 0.710485, 0.712766, 0.714263, 0.716545, 0.718826, 0.721107, 0.723388, 0.72567, 0.727951, 0.730232, 0.732514, 0.734795, 0.737076, 0.739358, 0.741639, 0.74392, 0.746202, 0.748483, 0.750764, 0.753045, 0.755327, 0.757608, 0.759889, 0.762171, 0.764452, 0.766733, 0.769015, 0.771296, 0.772793, 0.775074, 0.777356, 0.779637, 0.781918, 0.7842, 0.785697, 0.787978, 0.789475, 0.791756, 0.794038, 0.796319, 0.7986, 0.800882, 0.803163, 0.805444, 0.806941, 0.809223, 0.811504, 0.813785, 0.815282, 0.817564, 0.819845, 0.822126, 0.824408, 0.825905, 0.828186, 0.830467, 0.832749, 0.83503, 0.837311, 0.839593, 0.841874, 0.844155, 0.846436, 0.848718, 0.850999, 0.85328, 0.855562, 0.857843, 0.860124, 0.862406, 0.864687, 0.866968, 0.86925, 0.871531, 0.873812, 0.876093, 0.878375, 0.880656, 0.882937, 0.885219, 0.8875, 0.889781, 0.892063, 0.894344, 0.896625, 0.898907, 0.901188, 0.903469, 0.90575, 0.908032, 0.910313, 0.912594, 0.914876, 0.917157, 0.919438, 0.92172, 0.924001, 0.926282, 0.928564, 0.930845, 0.933126, 0.935407, 0.937689, 0.93997, 0.942251, 0.944533, 0.946814, 0.949095, 0.951377, 0.953658, 0.955939, 0.958221, 0.960502, 0.962783, 0.965064, 0.967346, 0.969627, 0.971908, 0.97419, 0.976471, 0.977968, 0.980249, 0.982531, 0.984812, 0.987093, 0.989375, 0.991656, 0.993937, 0.996219, 0.9985, 0.999997};
    std::vector<float> a = {1, 2, 3, 4, 5};
    std::vector<float> p = {10, 5, 3, 2, 1};
    int b = 1;

    std::vector<float> c = {};
    std::vector<float> d = {};
    std::vector<float> e = {};

    // convert_vec_to_float(a, b, c);
    // printv(c);
    normalize(p, d);
    printv(d);
    cumulative(d, e);
    printv(e);
    std::cout << "\n e size: "
              << e.size() << "\n";
    find_index_less_or_equal(e, 0);
    find_index_less_or_equal(e, 0.99999999);

    // for (int i = 0; i <= 10; i++)
    // {
    //     int n = 3;
    //     printv(choice_n(a, n, false, p));
    //     std::cout << '\n';
    // }

    return 0;
}

// 0.47619 0.238095 0.142857 0.0952381 0.047619
// 0.47619 0.714286 0.857143 0.952381 1
// n >= item
// 0>=0.47619
// 0

// n >= item
// 0.47 >= 0.47619
// 0

// n >= item
// 0.48 >= 0.714286
// 1

// n >= item
//1 >= 1
//4
