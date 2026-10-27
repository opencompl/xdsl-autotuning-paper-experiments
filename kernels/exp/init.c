#include <libxsmm_typedefs.h>
#include <stddef.h>

// Offsets used by the literal generator's System V unary ABI.
_Static_assert(offsetof(libxsmm_meltw_unary_param, in.primary) == 32,
               "Unexpected LIBXSMM unary input offset");
_Static_assert(offsetof(libxsmm_meltw_unary_param, out.primary) == 80,
               "Unexpected LIBXSMM unary output offset");

extern void exp_kernel(const libxsmm_meltw_unary_param *param);

// Return the assembly callback directly: no extra wrapper call in the timer.
libxsmm_meltwfunction_unary exp_init(void) { return exp_kernel; }
