/*
 * Defines a wrapper function for calling a TVM generated PackedFunc (type erased function working on dlpack tensor handles).
 *
 * Compile with, for instance with a generated matmul __tvm_main__ of size IxJxK == 256x1024x128:
 * gcc -O2 -DKERNEL_FUNC=matmul -DPACKED_FUNC=__tvm_main__ -DMM_I=256 -DMM_J=1024 -DMM_K=128 -c -o tvm_matmul_256_1024_128.o tvm_matmul_wrapper.c
 *
 * Note that the KERNEL_FUNC that will be defined is suppose to be:
 * extern void KERNEL_FUNC(const DTYPE *A, const DTYPE *B, DTYPE *C);
 *
 * Where A, B, C are continuous buffers of dimentsions A[MM_I, MM_K], B[MM_K, MM_J], C[MM_I, MM_J].
 *
 * Default DTYPE is float, to change to double pass -DMM_DTYPE=MM_DTYPE_double.
 *
 * Also the TVM packed function PACKED_FUNC is supposed to be generated with arguments orders A, B, C.
 *
 * Change below if arguments orders of KERNEL_FUNC and/or PACKED_FUNC differ.
 *
 * Ref below to "Parameters to be defined externally" for details of externally defined preprocessor constants.
 *
 * This implementation works for float only, though support for double could be easily added.
 *
 */      
#include <stdlib.h>
#include <stdint.h>


/* Parameters to be defined externally */
#ifndef KERNEL_FUNC
#error "Pass -DKERNEL_FUNC=<external_kernel_func> for the externally available bare ptr kernel function generated"
#endif
#ifndef PACKED_FUNC
#error "Pass -DPACKED_FUNC=<tvm_pack_func> for the packed function name"
#endif
#ifndef MM_I
#error "Pass -DMM_I=<I_size> for the I matmul dimension size"
#endif
#ifndef MM_J
#error "Pass -DMM_J=<J_size> for the J matmul dimension size"
#endif
#ifndef MM_K
#error "Pass -DMM_K=<K_size> for the K matmul dimension size"
#endif
/* If not specified, defaults to float. */
#ifndef MM_DTYPE
#define MM_DTYPE MM_DTYPE_float
#endif

/* Externally defined data type. */
#define MM_DTYPE_float 1
#define MM_DTYPE_double 2

#if MM_DTYPE == MM_DTYPE_float
#define DTYPE float
#else
#define DTYPE double
#endif

/* Refer to tvm c_runtime_api.h */
#define kTVMArgInt 0
#define kTVMDLTensorHandle 7

/* Refer to dlpack.h */
#define kDLCPU 1
#define kDLFloat 2

typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

/* Bridge bare pointers to TVM packed function with DLTensor objects */
#ifdef __cplusplus
extern "C"
#else
extern
#endif
void PACKED_FUNC(void *args[], const int32_t *args_types, int32_t num_args, void *res, const int32_t *res_types, void *resource_manager);

#ifdef __cplusplus
extern "C"
#endif
void KERNEL_FUNC(const DTYPE *A, const DTYPE *B, DTYPE *C) {
#if MM_DTYPE == MM_DTYPE_float
    static const DLDataType dtype = { kDLFloat, 32, 1 };
#else
    static const DLDataType dtype = { kDLFloat, 64, 1 };
#endif
    static const DLDevice dev = { kDLCPU, 0 };
    static const int64_t DL_A_shape[2] = { MM_I, MM_K };
    static const int64_t DL_B_shape[2] = { MM_K, MM_J };
    static const int64_t DL_C_shape[2] = { MM_I, MM_J };
    DLTensor DL_A = { (void *)A, dev, 2, dtype, (int64_t *)DL_A_shape, NULL, 0 };
    DLTensor DL_B = { (void *)B, dev, 2, dtype, (int64_t *)DL_B_shape, NULL, 0 };
    DLTensor DL_C = { (void *)C, dev, 2, dtype, (int64_t *)DL_C_shape, NULL, 0 };
    void *args[3] = { &DL_A, &DL_B, &DL_C };
    const int32_t types[3] = { kTVMDLTensorHandle, kTVMDLTensorHandle, kTVMDLTensorHandle };
    int64_t res;
    int32_t res_type = kTVMArgInt;
    PACKED_FUNC(args, types, 3, &res, &res_type, NULL);
}
