#include "test.h"
#include "alex.h"
#include "vgg.h"
#include "resnet.h"
#include "densenet.h"
#include "squeeze.h"
#include "mobile.h"
#include "mnasnet.h"
#include "inception.h"
#include "shuffle.h"
#include "efficient.h"
#include "regnet.h"
#include "origin_denoiser.h"
#include "custom_denoiser.h"
#include <cuda_profiler_api.h>

// #define n_dense 0
// #define n_res 0
// #define n_alex 0
// #define n_vgg 0
// #define n_wide 4
// #define n_squeeze 0
// #define n_mobile 0
// #define n_mnasnet 0
// // #define n_inception 0
// #define n_shuffle 0
// #define n_resX 0
// #define n_reg 0

#define n_threads 1 // inception 병렬실행하기 위한 최소한의 thread 갯수는 4개
#define WARMING 0

// index of flatten or gap
#define DENSE_FLATTEN 1
#define RES_FLATTEN 1   //WideResNet, ResNext
#define ALEX_FLATTEN 5
#define VGG_FLATTEN 5
#define SQUEEZE_FLATTEN 1
#define MOBILE_FLATTEN 1
#define MNAS_GAP 1
#define INCEPTION_FLATTEN 1
#define SHUFFLE_GAP 1
#define EFFICIENT_FLATTEN 1
#define REG_FLATTEN 1

// #define decompose

int total_iter;

extern void *predict_vgg(Net *input);
extern void *predict_resnet(Net *input);
extern void *predict_inception(Net *input);
extern void *predict_regnet(Net *input);
extern void *predict_origin_denoiser(Net *input);
extern void *predict_custom_denoiser(Net *input);
extern void *predict_decomposed_denoiser(Net *input);

extern void *predict_warm_inception(Net *input);
extern void *predict_warm_origin_denoiser(Net *input);
extern void *predict_warm_custom_denoiser(Net *input);
extern void *predict_warm_decomposed_denoiser(Net *input);

namespace F = torch::nn::functional;
using namespace std;

threadpool thpool;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;
// 2차원 배열형태로 사용하기위해 streams 벡터 다음과 같이 변경
std::vector<std::vector <at::cuda::CUDAStream>> streams;


c10::DeviceIndex GPU_NUM=0;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, const char* argv[]) {
  GPU_NUM=atoi(argv[1]);
  c10::cuda::set_device(GPU_NUM);
  torch::Device device = {at::kCUDA,GPU_NUM};

  // std::string filename = argv[2];


  int n_inception=atoi(argv[2]); //애네들 4명은 HIGH고 나머지는 싹다 LOW
  int n_alex = atoi(argv[3]);
  int n_vgg = atoi(argv[4]);
  int n_res = atoi(argv[5]);
  int n_dense = atoi(argv[6]);
  int n_wide = atoi(argv[7]);
  int n_squeeze = atoi(argv[8]);
  int n_mobile = atoi(argv[9]);
  int n_mnasnet = atoi(argv[10]);
  int n_shuffle = atoi(argv[11]);
  int n_resX = atoi(argv[12]);
  total_iter = atoi(argv[13]);



  
  int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX;
  int acc_index_n = 0; // 여기서 acc는 accumulate

  static int stream_index_L = 0;
  static int stream_index_H = 0;
  static int branch_index_L = 31;
  static int branch_index_H = 31;
  static int net_priority_L = n_all-1 -(n_inception); // LOW index model get HIHG priority
  static int net_priority_H = n_all-1;

  streams.resize(2); // streams[a][b], a=1이면 HIGH, a=0이면 LOW, b는 index값, HIGH,LOW각각 32개씩

   /* stream 생성 */
  for(int i=0; i<n_streamPerPool; i++){
    streams[1].push_back(at::cuda::getStreamFromPool(true,GPU_NUM)); //high priority stream  (stream priority 값 = -1)
  }
  for(int i=0; i<n_streamPerPool; i++){
    streams[0].push_back(at::cuda::getStreamFromPool(false,GPU_NUM)); //low priority stream  (stream priority 값 = 0)
  }

  #if FIFOQ
  thpool = thpool_init(n_threads);
  #else
  thpool = thpool_init(n_threads, n_all);
  #endif

  torch::jit::script::Module inceptionModule;
  torch::jit::script::Module denseModule;
  torch::jit::script::Module resModule;
  torch::jit::script::Module alexModule;
  torch::jit::script::Module vggModule;
  torch::jit::script::Module wideModule;
  torch::jit::script::Module squeezeModule;
  torch::jit::script::Module mobileModule;
  torch::jit::script::Module mnasModule;
  torch::jit::script::Module shuffleModule;
  torch::jit::script::Module resXModule;
  torch::jit::script::Module regModule;

  try {
      denseModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/densenet_model.pt");
      denseModule.to(device);

    	resModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/resnet_model.pt");
      resModule.to(device);

    	alexModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/alexnet_model.pt");
      alexModule.to(device);
  
    	vggModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/vgg_model.pt");
      vggModule.to(device);

    	wideModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/wideresnet_model.pt");
      wideModule.to(device);
 
    	squeezeModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/squeeze_model.pt");
      squeezeModule.to(device);

    	mobileModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/mobilenet_model.pt");
      mobileModule.to(device);

    	mnasModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/mnasnet_model.pt");
      mnasModule.to(device);

    	inceptionModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/inception_model.pt");
      inceptionModule.to(device);

    	shuffleModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/shuffle_model.pt");
      shuffleModule.to(device);

    	resXModule = torch::jit::load("/home/kmsjames/very-big-storage/JSH/joo_total/multi/resnext_model.pt");
      resXModule.to(device);


  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";



  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);


  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }


  vector<torch::jit::IValue> inputs;
  vector<torch::jit::IValue> inputs2;
  vector<torch::jit::IValue> inputs3;

  torch::Tensor x = torch::ones({1, 3, 224, 224}).to(device);
  inputs.push_back(x);

  torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(device);
  inputs3.push_back(x3);

  at::Tensor out;

  if(n_inception){
    torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(device);

    auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
      
    x_ch0.to(device);
    x_ch1.to(device);
    x_ch2.to(device);

    auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(device);
    inputs2.push_back(x_cat);
  }
  

  Net net_input_dense[n_dense];
  Net net_input_res[n_res];
  Net net_input_alex[n_alex];
  Net net_input_vgg[n_vgg];
  Net net_input_wide[n_wide];
  Net net_input_squeeze[n_squeeze];
  Net net_input_mobile[n_mobile];
  Net net_input_mnasnet[n_mnasnet];
  Net net_input_inception[n_inception];
  Net net_input_shuffle[n_shuffle];
  Net net_input_resX[n_resX];


  pthread_t networkArray_dense[n_dense];
  pthread_t networkArray_res[n_res];
  pthread_t networkArray_alex[n_alex];
  pthread_t networkArray_vgg[n_vgg];
  pthread_t networkArray_wide[n_wide];
  pthread_t networkArray_squeeze[n_squeeze];
  pthread_t networkArray_mobile[n_mobile];
  pthread_t networkArray_mnasnet[n_mnasnet];
  pthread_t networkArray_inception[n_inception];
  pthread_t networkArray_shuffle[n_shuffle];
  pthread_t networkArray_resX[n_resX];


  cout<<"***** Model Load compelete *****"<<"\n";


  for(int i=0;i<n_dense;i++){
    get_submodule_densenet(denseModule, net_input_dense[i]);
    std::cout << "End get submodule_densenet "<< i << "\n";
    net_input_dense[i].input = inputs;
    net_input_dense[i].name = "DenseNet";
    net_input_dense[i].flatten = net_input_dense[i].layers.size()-1;
    net_input_dense[i].index_n = i+acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_dense[i].H_L = 0; // stream priority의 default값은 low
    net_input_dense[i].index_s = stream_index_L;
    net_input_dense[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_dense[i].H_L = 0; 
    net_input_dense[i].index_s = stream_index_L;
    net_input_dense[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_densenet(&net_input_dense[i]);
      net_input_dense[i].input = inputs;
    }
    std::cout << "====== END DenseNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_dense;

  for(int i=0;i<n_res;i++){
    get_submodule_resnet(resModule, net_input_res[i]);
    std::cout << "End get submodule_resnet "<< i << "\n";
    net_input_res[i].input = inputs;
    net_input_res[i].name = "ResNet";
    net_input_res[i].flatten = net_input_res[i].layers.size()-1;
    net_input_res[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_res[i].H_L = 0; // stream priority의 default값은 low
    net_input_res[i].index_s = stream_index_L;
    net_input_res[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_res[i].H_L = 0; 
    net_input_res[i].index_s = stream_index_L;
    net_input_res[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_res[i]);
      net_input_res[i].input = inputs;
    }
    std::cout << "====== END ResNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_res;

  for(int i=0;i<n_wide;i++){
    get_submodule_resnet(wideModule, net_input_wide[i]);
    std::cout << "End get submodule_widenet "<< i << "\n";
    net_input_wide[i].input = inputs;
    net_input_wide[i].name = "WideResNet";
    net_input_wide[i].flatten = net_input_wide[i].layers.size() - RES_FLATTEN;
    net_input_wide[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_wide[i].H_L = 0; // stream priority의 default값은 low
    net_input_wide[i].index_s = stream_index_L;
    net_input_wide[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_wide[i].H_L = 0; 
    net_input_wide[i].index_s = stream_index_L;
    net_input_wide[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_wide[i]);
      net_input_wide[i].input = inputs;
    }
    std::cout << "====== END WideRes WARMUP ======" << std::endl;
  }
  acc_index_n += n_wide;

  for(int i=0;i<n_resX;i++){
    get_submodule_resnet(resXModule, net_input_resX[i]);
    std::cout << "End get submodule_resXnet "<< i << "\n";
    net_input_resX[i].input = inputs;
    net_input_resX[i].name = "ResNext";
    net_input_resX[i].flatten = net_input_resX[i].layers.size() - RES_FLATTEN;
    net_input_resX[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_resX[i].H_L = 0; // stream priority의 default값은 low
    net_input_resX[i].index_s = stream_index_L;
    net_input_resX[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_resX[i].H_L = 0; 
    net_input_resX[i].index_s = stream_index_L;
    net_input_resX[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_resnet(&net_input_resX[i]);
      net_input_resX[i].input = inputs;
    }
    std::cout << "====== END ResNext WARMUP ======" << std::endl;
  }
  acc_index_n += n_resX;

  for(int i=0;i<n_vgg;i++){
    get_submodule_vgg(vggModule, net_input_vgg[i]);
    std::cout << "End get submodule_vgg "<< i << "\n";
    net_input_vgg[i].input = inputs;
    net_input_vgg[i].name = "VGG";
    net_input_vgg[i].flatten = net_input_vgg[i].layers.size()- VGG_FLATTEN;
    net_input_vgg[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_vgg[i].H_L = 0; // stream priority의 default값은 low
    net_input_vgg[i].index_s = stream_index_L;
    net_input_vgg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_vgg[i].H_L = 0; 
    net_input_vgg[i].index_s = stream_index_L;
    net_input_vgg[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_vgg(&net_input_vgg[i]);
      net_input_vgg[i].input = inputs;
    }
    std::cout << "====== END VGG WARMUP ======" << std::endl;
  }
  acc_index_n += n_vgg;
  
  for(int i=0;i<n_inception;i++){
	  get_submodule_inception(inceptionModule, net_input_inception[i]);
    std::cout << "End get submodule_inception "<< i << "\n";
    for(int j=0;j<4;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_inception[i].record.push_back(event_temp);
    }
    net_input_inception[i].n_all = n_all;
	  net_input_inception[i].input = inputs2;
    net_input_inception[i].name = "Inception_v3";
    net_input_inception[i].flatten = net_input_inception[i].layers.size() - INCEPTION_FLATTEN;
    net_input_inception[i].index_n = i+acc_index_n;;
#if FIFOQ
    net_input_inception[i].H_L = 0; // stream priority의 default값은 low
    net_input_inception[i].index_s = stream_index_L;
    net_input_inception[i].index_b = branch_index_L;
    net_input_inception[i].priority = 0;
    stream_index_L +=1;
    branch_index_L -=3;
#else
    net_input_inception[i].H_L = 1; // HIGH priority stream
    net_input_inception[i].index_s = stream_index_H;
    net_input_inception[i].index_b = branch_index_H;
    net_input_inception[i].priority = net_priority_H;
    stream_index_H+=1;
    branch_index_H-=3;  
    // if(i%2 == 1)
    net_priority_H-=1;
#endif

    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_warm_inception(&net_input_inception[i]);
      net_input_inception[i].input = inputs2;
      for(int n=0;n<net_input_inception[i].layers.size();n++){
        net_input_inception[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_inception[i].fp = fopen((filename+"-"+"I"+".txt").c_str(),"a");
  }
  acc_index_n += n_inception;

  for(int i=0;i<n_alex;i++){
    get_submodule_alexnet(alexModule, net_input_alex[i]);
    std::cout << "End get submodule_alexnet "<< i << "\n";
    net_input_alex[i].input = inputs;
    net_input_alex[i].name = "AlexNet";
    net_input_alex[i].flatten = net_input_alex[i].layers.size()- ALEX_FLATTEN;
    net_input_alex[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_alex[i].H_L = 0; // stream priority의 default값은 low
    net_input_alex[i].index_s = stream_index_L;
    net_input_alex[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_alex[i].H_L = 0; 
    net_input_alex[i].index_s = stream_index_L;
    net_input_alex[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_alexnet(&net_input_alex[i]);
      net_input_alex[i].input = inputs;
    }
    std::cout << "====== END Alex WARMUP ======" << std::endl;
  }
  acc_index_n += n_alex;

  for(int i=0;i<n_squeeze;i++){
    get_submodule_squeeze(squeezeModule, net_input_squeeze[i]);
    std::cout << "End get submodule_squeezenet "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_squeeze[i].record.push_back(event_temp);
    }
    net_input_squeeze[i].input = inputs;
    net_input_squeeze[i].name = "SqueezeNet";
    net_input_squeeze[i].flatten = net_input_squeeze[i].layers.size()- SQUEEZE_FLATTEN;
    net_input_squeeze[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_squeeze[i].H_L = 0; // stream priority의 default값은 low
    net_input_squeeze[i].index_s = stream_index_L;
    net_input_squeeze[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_squeeze[i].H_L = 0; 
    net_input_squeeze[i].index_s = stream_index_L;
    net_input_squeeze[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_squeeze(&net_input_squeeze[i]);
      net_input_squeeze[i].input = inputs;
      for(int n=0;n<net_input_squeeze[i].layers.size();n++){
        net_input_squeeze[i].layers[n].exe_success = false;
      }
    }
    std::cout << "====== END Squeeze WARMUP ======" << std::endl;
  }
  acc_index_n += n_squeeze;

  for(int i=0;i<n_mobile;i++){
    get_submodule_mobilenet(mobileModule, net_input_mobile[i]);
    std::cout << "End get submodule_mobilenet "<< i << "\n";
    net_input_mobile[i].input = inputs;
    net_input_mobile[i].name = "Mobile";
    net_input_mobile[i].flatten = net_input_mobile[i].layers.size()- MOBILE_FLATTEN;
    net_input_mobile[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_mobile[i].H_L = 0; // stream priority의 default값은 low
    net_input_mobile[i].index_s = stream_index_L;
    net_input_mobile[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_mobile[i].H_L = 0; 
    net_input_mobile[i].index_s = stream_index_L;
    net_input_mobile[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_mobilenet(&net_input_mobile[i]);
      net_input_mobile[i].input = inputs;
    }
    std::cout << "====== END Mobile WARMUP ======" << std::endl;
  }
  acc_index_n += n_mobile;

  for(int i=0;i<n_mnasnet;i++){
    get_submodule_MNASNet(mnasModule, net_input_mnasnet[i]);
    std::cout << "End get submodule_mnasnet "<< i << "\n";
    net_input_mnasnet[i].input = inputs;
    net_input_mnasnet[i].name = "MNASNet";
    net_input_mnasnet[i].gap = net_input_mnasnet[i].layers.size() - MNAS_GAP;
    net_input_mnasnet[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_mnasnet[i].H_L = 0; // stream priority의 default값은 low
    net_input_mnasnet[i].index_s = stream_index_L;
    net_input_mnasnet[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_mnasnet[i].H_L = 0; 
    net_input_mnasnet[i].index_s = stream_index_L;
    net_input_mnasnet[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_MNASNet(&net_input_mnasnet[i]);
      net_input_mnasnet[i].input = inputs;
    }
    std::cout << "====== END mnasnet WARMUP ======" << std::endl;
  }
  acc_index_n += n_mnasnet;

  for(int i=0;i<n_shuffle;i++){
    get_submodule_shuffle(shuffleModule, net_input_shuffle[i]);
    std::cout << "End get submodule_shuffle "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_shuffle[i].record.push_back(event_temp);
    }
    net_input_shuffle[i].input = inputs;
    net_input_shuffle[i].name = "ShuffleNet";
    net_input_shuffle[i].gap = net_input_shuffle[i].layers.size() - SHUFFLE_GAP;
    net_input_shuffle[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_shuffle[i].H_L = 0; // stream priority의 default값은 low
    net_input_shuffle[i].index_s = stream_index_L;
    net_input_shuffle[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_shuffle[i].H_L = 0; 
    net_input_shuffle[i].index_s = stream_index_L;
    net_input_shuffle[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_shuffle(&net_input_shuffle[i]);
      net_input_shuffle[i].input = inputs;
      for(int n=0;n<net_input_shuffle[i].layers.size();n++){
        net_input_shuffle[i].layers[n].exe_success = false;
      }
    }
    std::cout << "====== END shuffle WARMUP ======" << std::endl;
  }
  acc_index_n += n_shuffle;

  std::cout<<"\n==================WARM UP END==================\n";
  // cudaDeviceSynchronize();
  
  // cudaProfilerStart();

  cudaEvent_t t_start, t_end;
  float t_time;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  
  //double time1 = what_time_is_it_now();

  for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &net_input_dense[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &net_input_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_alex;i++){
    if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_vgg;i++){
	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_wide;i++){
    if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))predict_resnet, &net_input_wide[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_squeeze;i++){
    if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mobile;i++){
    if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mnasnet;i++){
    if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))predict_MNASNet, &net_input_mnasnet[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_inception;i++){
    if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_shuffle;i++){
    if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))predict_shuffle, &net_input_shuffle[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_resX;i++){
    if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))predict_resnet, &net_input_resX[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  

  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL); // pthread_join : thread 종료를 기다리고 thread 종료 이후 다음 진행
  }                                            // join된 thread(종료된 thread)는 모든 resource를 반납

    for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }

  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }

  for (int i = 0; i < n_vgg; i++){
    pthread_join(networkArray_vgg[i], NULL);
  }

  for (int i = 0; i < n_wide; i++){
    pthread_join(networkArray_wide[i], NULL);
  }

  for (int i = 0; i < n_squeeze; i++){
    pthread_join(networkArray_squeeze[i], NULL);
  }

  for (int i = 0; i < n_mobile; i++){
    pthread_join(networkArray_mobile[i], NULL);
  }

  for (int i = 0; i < n_mnasnet; i++){
    pthread_join(networkArray_mnasnet[i], NULL);
  }

  for (int i = 0; i < n_inception; i++){
    pthread_join(networkArray_inception[i], NULL);
  }

  for (int i = 0; i < n_shuffle; i++){
    pthread_join(networkArray_shuffle[i], NULL);
  }

  for (int i = 0; i < n_resX; i++){
    pthread_join(networkArray_resX[i], NULL);
  }


  //cudaDeviceSynchronize();
  //double time2 = what_time_is_it_now();

  cudaDeviceSynchronize();
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&t_time, t_start, t_end);

	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
  // cudaProfilerStop();
  free(cond_t);
  free(mutex_t);
  free(cond_i);
}