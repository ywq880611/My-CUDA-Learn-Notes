#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


// Elementwise add FP32
// grid: N/256, block: 256
// shape of a, b and c: (N)
__global__ void elementwise_add_f32_kernel(float* a, float* b, float* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = b[idx] + a[idx];
    }
}

// Elementwise add FP32*4
// grid: N/256, block: 256/4
// shape of a, b and c: (N)
__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4 aa = FLOAT4(a[idx]);
        float4 bb = FLOAT4(b[idx]);
        float4 cc;
        cc.x = aa.x + bb.x;
        cc.y = aa.y + bb.y;
        cc.z = aa.z + bb.z;
        cc.w = aa.w + bb.w;
        FLOAT4(c[idx]) = cc;
    }
}

// -------------------------------------- FP16 -------------------------------------- 
// ElementWise Add  
// grid(N/256), block(256)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int N){
  int idx = blockIdx.x * blockDim.x + + threadIdx.x;
  if(idx < N){
    c[idx] = a[idx] + b[idx];
  }
}

// ElementWise Add  
// grid(N/256), block(256)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_hadd_kernel(half* a, half* b, half* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = __hadd(a[idx], b[idx]);
}

// ElementWise Add * 2
// grid(N/256), block(256/2)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half* a, half* b, half* c, int N){
  int idx = 2 * (blockIdx.x * blockDim.x + + threadIdx.x);
  if(idx < N){
    half2 aa = HALF2(a[idx]);
    half2 bb = HALF2(b[idx]);
    half2 cc;
    cc.x = aa.x + bb.x;
    cc.y = aa.y + bb.y;
    HALF2(c[idx]) = cc;
  }
}

// ElementWise Add * 8
// grid(N/256), block(256/8)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x8_kernel(half* a, half* b, half* c, int N){
  int idx = 8 * (blockIdx.x * blockDim.x + + threadIdx.x);
  if(idx < N){
    half2 aa1 = HALF2(a[idx]);
    half2 bb1 = HALF2(b[idx]);
    half2 aa2 = HALF2(a[idx + 2]);
    half2 bb2 = HALF2(b[idx + 2]);
    half2 aa3 = HALF2(a[idx + 4]);
    half2 bb3 = HALF2(b[idx + 4]);
    half2 aa4 = HALF2(a[idx + 6]);
    half2 bb4 = HALF2(b[idx + 6]);
    half2 cc1, cc2, cc3, cc4;
    cc1.x = aa1.x + bb1.x;
    cc1.y = aa1.y + bb1.y;
    cc2.x = aa2.x + bb2.x;
    cc2.y = aa2.y + bb2.y;
    cc3.x = aa3.x + bb3.x;
    cc3.y = aa3.y + bb3.y;
    cc4.x = aa4.x + bb4.x;
    cc4.y = aa4.y + bb4.y;
    HALF2(c[idx]) = cc1;
    HALF2(c[idx + 2]) = cc2;
    HALF2(c[idx + 4]) = cc3;
    HALF2(c[idx + 6]) = cc4;
  }
}

// ElementWise Add * 8
// grid(N/256), block(256/8)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x8_pack_kernel(half* a, half* b, half* c, int N){
  int idx = 8 * (blockIdx.x * blockDim.x + + threadIdx.x);
  half pack_a[8], pack_b[8], pack_c[8];
  // assign pack_a point to a[idx] and reinterpret as float4
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

  for(int i = 0; i < 8; i += 2){
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }

  if((idx + 7) < N){
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  }
}


// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements)   \
void elementwise_add_##packed_type(                                              \
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {                           \
  CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                         \
  const int ndim = a.dim();                                                      \
  if (ndim != 2) {                                                               \
    int N = 1;                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= a.size(i); }                           \
    dim3 block(256 / (n_elements));                                              \
    dim3 grid((N + 256 - 1) / 256);                                              \
    elementwise_add_##packed_type##_kernel<<<grid, block>>>(                     \
      reinterpret_cast<element_type*>(a.data_ptr()),                             \
      reinterpret_cast<element_type*>(b.data_ptr()),                             \
      reinterpret_cast<element_type*>(c.data_ptr()), N);                         \
  } else {                                                                       \
    const int S = a.size(0);                                                     \
    const int K = a.size(1);                                                     \
    const int N = S * K;                                                         \
    if ((K/(n_elements)) <= 1024) {                                              \
      dim3 block(K/(n_elements));                                                \
      dim3 grid(S);                                                              \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                   \
        reinterpret_cast<element_type*>(a.data_ptr()),                           \
        reinterpret_cast<element_type*>(b.data_ptr()),                           \
        reinterpret_cast<element_type*>(c.data_ptr()), N);                       \
    } else {                                                                     \
      int N = 1;                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= a.size(i); }                         \
      dim3 block(256 / (n_elements));                                            \
      dim3 grid((N + 256 - 1) / 256);                                            \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                   \
        reinterpret_cast<element_type*>(a.data_ptr()),                           \
        reinterpret_cast<element_type*>(b.data_ptr()),                           \
        reinterpret_cast<element_type*>(c.data_ptr()), N);                       \
    }                                                                            \
  }                                                                              \
}


TORCH_BINDING_ELEM_ADD(f32,         torch::kFloat32,    float,    1)
TORCH_BINDING_ELEM_ADD(f32x4,       torch::kFloat32,    float,    4)
TORCH_BINDING_ELEM_ADD(f16,         torch::kHalf,       half,     1)
TORCH_BINDING_ELEM_ADD(f16_hadd,    torch::kHalf,       half,     1)
TORCH_BINDING_ELEM_ADD(f16x2,       torch::kHalf,       half,     2)
TORCH_BINDING_ELEM_ADD(f16x8,       torch::kHalf,       half,     8)
TORCH_BINDING_ELEM_ADD(f16x8_pack,  torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16_hadd)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}