#include <vector>
#include <cmath>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/util/complex.h>

#include <ATen/ATen.h>

#define threadnum 8

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {

  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template <typename scalar_t>
__global__ void FHtoInt8_kernel(Pack<scalar_t, 8>* input, Pack<int8_t, 8>* output, float idx_value, float scale) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  Pack<scalar_t, 8> data = input[idx];
  Pack<int8_t, 8> result;

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    scalar_t sign = (data.elem[i] >= 0) ? 1 : -1;
    result.elem[i] = static_cast<int8_t>(sign * scale * pow(abs(data.elem[i]), 1.0f / idx_value));
  }
  __syncthreads();
  output[idx] = result;
}

template <typename scalar_t>
__global__ void Int8ToFH_kernel(Pack<int8_t, 8>* input, Pack<scalar_t, 8>* output, float idx_value, float scale) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  Pack<int8_t, 8> data = input[idx];
  Pack<scalar_t, 8> result ;
    
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    scalar_t half_data = static_cast<scalar_t>(data.elem[i]);
    scalar_t sign = (half_data >= 0) ? 1 : -1;
    scalar_t output = sign * pow((1/scale) * abs(half_data), idx_value);
    result.elem[i] = output;
  }
  __syncthreads();
  output[idx] = result;
}


// from float32/half to int8
void FHtoInt8(torch::Tensor& tensor_in, torch::Tensor& tensor_out , float idx_value, float scale) {
    //开启线程数量
    long size = tensor_in.size(0) / 8 ;
    // 计算 grid 大小
    long blockDIM = 256;
    int gridDIM = size / blockDIM;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_in.type(), "FHtoInt8_kernel_cuda", ([&] {
      // 调用 CUDA kernel
      FHtoInt8_kernel<scalar_t><<<gridDIM, blockDIM>>>(
                                      reinterpret_cast<Pack<scalar_t, 8>*>(tensor_in.data_ptr()), 
                                      reinterpret_cast<Pack<int8_t, 8>*>(tensor_out.data_ptr()), 
                                      idx_value, 
                                      scale);
    }));
}

// from int8 to float32 or half
void Int8ToFH(torch::Tensor& tensor_in, torch::Tensor& tensor_out , float idx_value, float scale) {
   
  //开启线程数量
  long size = tensor_in.size(0) / 8;
  // 计算 grid 大小
  long blockDIM = 256;
  int gridDIM = size / blockDIM;
  // 调用 CUDA kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_out.type(), "FHtoInt8_kernel_cuda", ([&] {
      // 调用 CUDA kernel
      Int8ToFH_kernel<scalar_t><<<gridDIM, blockDIM>>>(
                                      reinterpret_cast<Pack<int8_t, 8>*>(tensor_in.data_ptr()), 
                                      reinterpret_cast<Pack<scalar_t, 8>*>(tensor_out.data_ptr()), 
                                      idx_value, 
                                      scale);
    }));
}

// template <typename scalar_t>
__global__ void Float2Int4_kernel(Pack<float, threadnum>* in_tensor, float* scalesF, float* zerosF, 
                                  Pack<unsigned char, threadnum/2>* uint8_out,int groupsize) {
    long index = (threadIdx.x) + static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x);
    Pack<float, threadnum> data = in_tensor[index];
    extern __shared__ float bufferF[];
    float* max_input = bufferF;
    float* min_input = bufferF + blockDim.x;
    float* scales_input = bufferF + blockDim.x*2;
    float* zeros_input = bufferF + blockDim.x*2 + (blockDim.x*threadnum)/groupsize;
    float max_local = data.elem[0];
    float min_local = data.elem[0];
    #pragma unroll
    for (int i = 1; i < threadnum; i++) {
        max_local = max(max_local, data.elem[i]);
        min_local = min(min_local, data.elem[i]);
    }
    max_input[threadIdx.x] = max_local ;
    min_input[threadIdx.x] = min_local ;
    __syncthreads();
    if(threadIdx.x < (blockDim.x*threadnum / groupsize) ){
        // nv samle   cub库
        int length = groupsize/threadnum; // for shared data
        float max_value = max_input[threadIdx.x * length];
        float min_value = min_input[threadIdx.x * length];
        #pragma unroll
        for (int i = threadIdx.x * length + 1; i < threadIdx.x * length + length; i++) {
          max_value = max(max_value, max_input[i]);
          min_value = min(min_value, min_input[i]);
        }
        float scales_ = (max_value - min_value) / 15;
        float zeros_ = min_value + scales_ * 8;

        scales_input[threadIdx.x] = scales_;
        zeros_input[threadIdx.x] = zeros_;
        // scales_input,zeros_input ->global
        scalesF[(blockDim.x * threadnum/groupsize) * blockIdx.x + threadIdx.x ] = scales_;
        zerosF[(blockDim.x * threadnum/groupsize) * blockIdx.x + threadIdx.x] = zeros_;
        
    }
    __syncthreads();
    // groupid为组号,拿到scales_和zeros_
    int groupid = threadIdx.x / (groupsize/threadnum);
    float scales_ = scales_input[groupid];
    float zeros_ = zeros_input[groupid];
    unsigned char out[threadnum];
    #pragma unroll
    for (int i = 0; i < threadnum; i++) {
        // 可加pow
        out[i] = static_cast<unsigned char>(std::clamp(static_cast<int>(std::round((data.elem[i] - (zeros_- 8*scales_)) / scales_)),0 ,15));
    }
    #pragma unroll
    for (int i = 1; i < threadnum; i += 2) {
        out[i] = out[i] << 4;
    }
   
    Pack<unsigned char, threadnum/2> temp;
    #pragma unroll
    for (int i = 0; i < threadnum/2; i += 1) {
        temp.elem[i] = out[2*i] + out[2*i+1];
    }
    uint8_out[index] = temp;
}


__global__ void Half2Int4_kernel(Pack<c10::Half, threadnum>* in_tensor, c10::Half* scales, c10::Half* zeros, Pack<unsigned char, threadnum/2>* uint8_out,int groupsize) {
    long index = (threadIdx.x) + static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x);
    Pack<c10::Half, threadnum> data = in_tensor[index];
    extern __shared__ c10::Half bufferH[];
    c10::Half* max_input = bufferH;
    c10::Half* min_input = bufferH + blockDim.x;
    c10::Half* scales_input = bufferH + blockDim.x*2;
    c10::Half* zeros_input = bufferH + blockDim.x*2 + (blockDim.x*threadnum)/groupsize;
    c10::Half max_local = data.elem[0];
    c10::Half min_local = data.elem[0];
    #pragma unroll
    for (int i = 1; i < threadnum; i++) {
        max_local = max(max_local, data.elem[i]);
        min_local = min(min_local, data.elem[i]);
    }
    max_input[threadIdx.x] = max_local ;
    min_input[threadIdx.x] = min_local ;
    __syncthreads();
    //#pragma unroll
    if(threadIdx.x < (blockDim.x*threadnum / groupsize) ){
        // nv samle   cub库
        int length = groupsize/threadnum; // for shared data
        c10::Half max_value = max_input[threadIdx.x * length];
        c10::Half min_value = min_input[threadIdx.x * length];
        #pragma unroll
        for (int i = threadIdx.x * length + 1; i < threadIdx.x * length + length; i++) {
          max_value = max(max_value, max_input[i]);
          min_value = min(min_value, min_input[i]);
        }
        c10::Half scales_ = (max_value - min_value) / 15;
        c10::Half zeros_ = min_value + scales_ * 8;

        scales_input[threadIdx.x] = scales_;
        zeros_input[threadIdx.x] = zeros_;
        // scales_input,zeros_input ->global
        scales[(blockDim.x * threadnum/groupsize) * blockIdx.x + threadIdx.x ] = scales_;
        zeros[(blockDim.x * threadnum/groupsize) * blockIdx.x + threadIdx.x] = zeros_;
        
    }

    __syncthreads();
    // groupid为组号,拿到scales_和zeros_
    int groupid = threadIdx.x / (groupsize/threadnum);
    c10::Half scales_ = scales_input[groupid];
    c10::Half zeros_ = zeros_input[groupid];
    unsigned char out[threadnum];
    #pragma unroll
    for (int i = 0; i < threadnum; i++) {
        // 可加pow
        out[i] = static_cast<unsigned char>(std::clamp(static_cast<int>(std::round((data.elem[i] - (zeros_- 8*scales_)) / scales_)),0 ,15));
    }
    #pragma unroll
    for (int i = 1; i < threadnum; i += 2) {
        out[i] = out[i] << 4;
    }
   
    Pack<unsigned char, threadnum/2> temp;
    #pragma unroll
    for (int i = 0; i < threadnum/2; i += 1) {
        temp.elem[i] = out[2*i] + out[2*i+1];
    }
    uint8_out[index] = temp;
}
// from float32/half to int4
void FHtoInt4(torch::Tensor& in_tensor, long n , torch::Tensor& scales , 
                torch::Tensor& zeros, torch::Tensor& uint8_out, int groupsize) {
    //开启线程数量
    // 计算 grid 大小
    long blockDIM = 0;
    long gridDIM = 0;

    switch (in_tensor.type().scalarType()) {
      case torch::ScalarType::Float:
            blockDIM = 256;
            gridDIM = static_cast<long>(pow(2, (n-8-log2(threadnum))));
            Float2Int4_kernel<<<gridDIM, blockDIM, (blockDIM*2 + (blockDIM*threadnum*2)/groupsize)*4>>>(
                                      reinterpret_cast<Pack<float, threadnum>*>(in_tensor.data_ptr()), 
                                      reinterpret_cast<float*>(scales.data_ptr()), 
                                      reinterpret_cast<float*>(zeros.data_ptr()),  
                                      reinterpret_cast<Pack<unsigned char, threadnum/2>*>(uint8_out.data_ptr()),
                                      groupsize);
            break;
      case torch::ScalarType::Half:
            blockDIM = 256;
            gridDIM = static_cast<long>(pow(2, (n-8-log2(threadnum))));
            Half2Int4_kernel<<<gridDIM, blockDIM, (blockDIM*2 + (blockDIM*threadnum*2)/groupsize)*sizeof(half)>>>(
                                        reinterpret_cast<Pack<c10::Half, threadnum>*>(in_tensor.data_ptr()), 
                                        reinterpret_cast<c10::Half*>(scales.data_ptr()), 
                                        reinterpret_cast<c10::Half*>(zeros.data_ptr()), 
                                        reinterpret_cast<Pack<unsigned char, threadnum/2>*>(uint8_out.data_ptr()),
                                        groupsize);
            break;
      default:
        break;
    }
}

template <typename scalar_t>
__global__ void Int4ToFH_kernel(scalar_t* scales, scalar_t* zeros,  Pack<unsigned char, threadnum/2>* uint8_out, Pack<scalar_t, threadnum>* out_tensor,int groupsize) {
    long index = static_cast<long>(threadIdx.x) + static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x);
    Pack<unsigned char, threadnum/2> num = uint8_out[index];
    scalar_t data[threadnum];
    #pragma unroll
    for (int i = 0; i < threadnum/2; i+=1) {
        data[2*i] = static_cast<scalar_t>(num.elem[i] & 0xF);
        data[2*i+1] = static_cast<scalar_t>((num.elem[i] >> 4) & 0xF);
    }
    int groupid = (blockDim.x * threadnum/groupsize) * blockIdx.x + threadIdx.x / (groupsize/threadnum);
    scalar_t scales_ = scales[groupid];
    scalar_t zeros_ = zeros[groupid];
    // dequantizite
    #pragma unroll
    for (int i = 0; i < threadnum; i++) {
          data[i] =(data[i] - 8) * scales_ + zeros_ ;
    }
    Pack<scalar_t, threadnum> temp;
    #pragma unroll
    for (int i = 0; i < threadnum; i++) {
        temp.elem[i] = data[i];
    }
    out_tensor[index] = temp;
}

// from int8 to float32 or half
void Int4ToFH(torch::Tensor& scales , torch::Tensor& zeros, torch::Tensor& uint8_out, 
              long n, torch::Tensor& out_tensor ,int groupsize) {
    // 计算 grid 大小
    long blockDIM = 256;
    long gridDIM = static_cast<long>(pow(2, (n-8-log2(threadnum))));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out_tensor.type(), "Int4ToFH_kernel_cuda", ([&] {
    // 调用 CUDA kernel
    Int4ToFH_kernel<scalar_t><<<gridDIM, blockDIM>>>(
                                      reinterpret_cast<scalar_t*>(scales.data_ptr()), 
                                      reinterpret_cast<scalar_t*>(zeros.data_ptr()), 
                                      reinterpret_cast< Pack<unsigned char, threadnum/2>*>(uint8_out.data_ptr()), 
                                      reinterpret_cast<Pack<scalar_t, threadnum>*>(out_tensor.data_ptr()),
                                      groupsize);
    }));
    }

__global__ void Float2Half_kernel(Pack<float, 4>* input, Pack<half, 4>* output, float idx_value, float scale) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  Pack<float, 4> data = input[idx];
  Pack<half, 4> result;

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    if (idx_value == 0){
        result.elem[i] = __float2half(scale * data.elem[i]);
    }else{
        float sign = (data.elem[i] >= 0) ? 1 : -1;
        result.elem[i] = __float2half(sign * scale * pow(abs(data.elem[i]), 1.0f / idx_value));
    }
    
  }
  __syncthreads();
  output[idx] = result;
}

void Float2Half(torch::Tensor& tensor_in, torch::Tensor& tensor_out , float idx_value, float scale) {
    //开启线程数量
    long size = tensor_in.size(0) / 4;
    // 计算 grid 大小
    long blockDIM = 256;
    int gridDIM = size / blockDIM;

    Float2Half_kernel<<<gridDIM, blockDIM>>>(
                                    reinterpret_cast<Pack<float, 4>*>(tensor_in.data_ptr()), 
                                    reinterpret_cast<Pack<half, 4>*>(tensor_out.data_ptr()), 
                                    idx_value, 
                                    scale);
}


__global__ void Half2Float_kernel(Pack<half, 4>* input, Pack<float, 4>* output, float idx_value, float scale) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  Pack<half, 4> data = input[idx];
  Pack<float, 4> result;
    
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float half_data = __half2float(data.elem[i]);
    if (idx_value == 0){
      result.elem[i] = (1/scale) *  half_data;
    }else{
      float sign = (half_data >= 0) ? 1 : -1;
      result.elem[i] = sign * pow((1/scale) * abs(half_data), idx_value);
    }
  }
  __syncthreads();
  output[idx] = result;
}

// from int8 to float32 or half
void Half2Float(torch::Tensor& tensor_in, torch::Tensor& tensor_out , float idx_value, float scale) {

  //开启线程数量
  long size = tensor_in.size(0) / 4;
  // 计算 grid 大小
  long blockDIM = 256;
  int gridDIM = size / blockDIM;
  // 调用 CUDA kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_out.type(), "FHtoInt8_kernel_cuda", ([&] {
      // 调用 CUDA kernel
      Half2Float_kernel<<<gridDIM, blockDIM>>>(
                                      reinterpret_cast<Pack<half, 4>*>(tensor_in.data_ptr()), 
                                      reinterpret_cast<Pack<float, 4>*>(tensor_out.data_ptr()), 
                                      idx_value, 
                                      scale);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Float2Half", &Float2Half, "CUDA kernel: from float to half");
    m.def("Half2Float", &Half2Float, "CUDA kernel: from half to float ");
    m.def("FHtoInt8", &FHtoInt8, "CUDA kernel: from float32/half to int8");
    m.def("Int8ToFH", &Int8ToFH, "CUDA kernel: from int8 to float32/half");
    m.def("FHtoInt4", &FHtoInt4, "CUDA kernel: from float32/half to int4");
    m.def("Int4ToFH", &Int4ToFH, "CUDA kernel: from int4 to float32/half");
}
