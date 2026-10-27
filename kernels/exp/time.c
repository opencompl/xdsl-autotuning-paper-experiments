#include <inttypes.h>
#include <libxsmm.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "exp.h"
#include "../../headers/perf.h"

#define ITERATIONS (268435456 / LENGTH)

extern void exp_kernel(const libxsmm_meltw_unary_param *param);
extern const unsigned char exp_kernel_end[];
extern void compxsmm_manual_exp_kernel(const libxsmm_meltw_unary_param *param);
extern const unsigned char compxsmm_manual_exp_kernel_end[];
extern void compxsmm_exp_kernel(const libxsmm_meltw_unary_param *param);
extern const unsigned char compxsmm_exp_kernel_end[];

// One call site for all variants, placements, and both execution orders.
__attribute__((noinline)) static TIMETY measure(
    libxsmm_meltwfunction_unary kernel, const libxsmm_meltw_unary_param *param) {
  for (int i = 0; i < 1000; ++i) kernel(param);
  time_start();
  for (int i = 0; i < ITERATIONS; ++i) kernel(param);
  return time_end(FREQ) / ITERATIONS;
}

int main(void) {
  libxsmm_meltwfunction_unary native = exp_init();
  if (!native) return 1;
  libxsmm_kernel_info info = {0};
  if (libxsmm_get_kernel_info((const void *)native, &info) != 0) return 1;
  const struct {
    const char *name;
    libxsmm_meltwfunction_unary kernel;
    size_t size;
  } variants[] = {
      {"libxsmm", native, info.code_size},
      {"xdsl_libxsmm", exp_kernel,
       (uintptr_t)exp_kernel_end - (uintptr_t)exp_kernel},
      {"compxsmm_manual", compxsmm_manual_exp_kernel,
       (uintptr_t)compxsmm_manual_exp_kernel_end - (uintptr_t)compxsmm_manual_exp_kernel},
      {"compxsmm", compxsmm_exp_kernel,
       (uintptr_t)compxsmm_exp_kernel_end - (uintptr_t)compxsmm_exp_kernel},
  };
  enum { VARIANTS = sizeof(variants) / sizeof(*variants) };
  const size_t offsets[] = {EXP_PLACEMENTS};
  const size_t page = (size_t)sysconf(_SC_PAGESIZE);
  size_t capacity = 0;
  for (int v = 0; v < VARIANTS; ++v)
    if (variants[v].size > capacity) capacity = variants[v].size;
  capacity = (capacity + offsets[sizeof(offsets) / sizeof(*offsets) - 1]
              + page - 1) / page * page;
  unsigned char *buffer = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buffer == MAP_FAILED) {
    perror("allocating exp code buffer");
    return 1;
  }

  _Alignas(64) float input[LENGTH + 32], output[LENGTH + 32];
  float saved[LENGTH + 32], expected[VARIANTS][LENGTH];
  for (int i = 0; i < LENGTH + 32; ++i) input[i] = -12345.0f;
  fill_exp_input(input + 16, 42);
  memcpy(saved, input, sizeof(input));
  libxsmm_meltw_unary_param param = {0};
  param.in.primary = input + 16;
  for (int variant = 0; variant < VARIANTS; ++variant) {
    param.out.primary = expected[variant];
    variants[variant].kernel(&param);
  }
  param.out.primary = output + 16;
  time_init();

  // Keep every placement and forward/reverse orders. No alignment search.
  printf("[");
  const char *separator = "";
  for (size_t p = 0; p < sizeof(offsets) / sizeof(*offsets); ++p) {
    unsigned char *code = buffer + offsets[p];
    for (int order = 0; order < 2; ++order) {
      for (int position = 0; position < VARIANTS; ++position) {
        int variant = order ? VARIANTS - 1 - position : position;
        // The kernels contain only local branches and inline RIP-relative
        // constants, so copying preserves their references. Never map RWX.
        if (mprotect(buffer, capacity, PROT_READ | PROT_WRITE)) goto protection_error;
        memset(buffer, 0xcc, capacity);
        memcpy(code, (const void *)variants[variant].kernel, variants[variant].size);
        __builtin___clear_cache((char *)code, (char *)code + variants[variant].size);
        if (mprotect(buffer, capacity, PROT_READ | PROT_EXEC)) goto protection_error;
        for (int i = 0; i < LENGTH + 32; ++i) output[i] = -12345.0f;
        TIMETY elapsed = measure((libxsmm_meltwfunction_unary)code, &param);

        // Check the relocated code against its original, outside timing.
        if (memcmp(output + 16, expected[variant], sizeof(expected[variant])) ||
            memcmp(input, saved, sizeof(input))) {
          fputs("Relocated exp changed its output or input.\n", stderr);
          munmap(buffer, capacity);
          return 1;
        }
        for (int i = 0; i < 16; ++i) {
          if (output[i] != -12345.0f || output[LENGTH + 16 + i] != -12345.0f) {
            fputs("Relocated exp overwrote an output guard.\n", stderr);
            munmap(buffer, capacity);
            return 1;
          }
        }
        printf("%s{\"variant\":\"%s\",\"placement_offset\":%zu,"
               "\"order\":%d,\"code_address\":\"0x%" PRIxPTR "\",\"time\":%Lf}",
               separator, variants[variant].name, offsets[p], order, (uintptr_t)code, elapsed);
        separator = ",";
      }
    }
  }
  puts("]");
  return munmap(buffer, capacity) != 0;

protection_error:
  perror("protecting exp code buffer");
  munmap(buffer, capacity);
  return 1;
}
