#ifndef AMX_H
#define AMX_H

// Linux keeps AMX tile data off until a process asks for it, and faults the
// first tile instruction of a process that has not; a driver of AMX kernels
// asks once, before the first call.  Without AMX this does nothing.
#ifdef AMX
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static inline void request_amx(void) {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
    perror("arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)");
    exit(EXIT_FAILURE);
  }
}
#else
static inline void request_amx(void) {}
#endif

#endif // AMX_H
