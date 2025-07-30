# RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t kernels/matmul_colmaj/test.c %s && %t | filecheck %s

        .file   "libxsmm.f64.x86.c"
        .text
        .globl  matmul_colmaj                      # -- Begin function matmul_colmaj
        .p2align        4
        .type   matmul_colmaj,@function
matmul_colmaj:                             # @matmul_colmaj
        .cfi_startproc
# %bb.0:
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        pushq   %r12
        .cfi_offset %r12, -24
        movq    %rdi, -16(%rbp)
        movq    %rsi, -24(%rbp)
        movq    %rdx, -32(%rbp)
        #APP
        movq    -16(%rbp), %rdi
        movq    -24(%rbp), %rsi
        movq    -32(%rbp), %rdx
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $192, %rsp
        movq    $-64, %r10
        andq    %r10, %rsp
        subq    $64, %rsp
        subq    $64, %rsp
        movq    $0, %r11
.Ltmp0:
        addq    $2, %r11
        movq    $0, %r10
.Ltmp1:
        addq    $4, %r10
        vmovupd (%rdx), %ymm14
        vmovupd 32(%rdx), %ymm15
        vmovupd (%rdi), %ymm2
        addq    $32, %rdi
        vbroadcastsd    (%rsi), %ymm0
        vfmadd231pd     %ymm2, %ymm0, %ymm14    # ymm14 = (ymm0 * ymm2) + ymm14
        vbroadcastsd    32(%rsi), %ymm1
        addq    $8, %rsi
        vfmadd231pd     %ymm2, %ymm1, %ymm15    # ymm15 = (ymm1 * ymm2) + ymm15
        vmovupd (%rdi), %ymm2
        addq    $32, %rdi
        vbroadcastsd    (%rsi), %ymm0
        vfmadd231pd     %ymm2, %ymm0, %ymm14    # ymm14 = (ymm0 * ymm2) + ymm14
        vbroadcastsd    32(%rsi), %ymm1
        addq    $8, %rsi
        vfmadd231pd     %ymm2, %ymm1, %ymm15    # ymm15 = (ymm1 * ymm2) + ymm15
        vmovupd (%rdi), %ymm2
        addq    $32, %rdi
        vbroadcastsd    (%rsi), %ymm0
        vfmadd231pd     %ymm2, %ymm0, %ymm14    # ymm14 = (ymm0 * ymm2) + ymm14
        vbroadcastsd    32(%rsi), %ymm1
        addq    $8, %rsi
        vfmadd231pd     %ymm2, %ymm1, %ymm15    # ymm15 = (ymm1 * ymm2) + ymm15
        vmovupd (%rdi), %ymm2
        addq    $32, %rdi
        vbroadcastsd    (%rsi), %ymm0
        vfmadd231pd     %ymm2, %ymm0, %ymm14    # ymm14 = (ymm0 * ymm2) + ymm14
        vbroadcastsd    32(%rsi), %ymm1
        addq    $8, %rsi
        vfmadd231pd     %ymm2, %ymm1, %ymm15    # ymm15 = (ymm1 * ymm2) + ymm15
        subq    $32, %rsi
        vmovupd %ymm14, (%rdx)
        vmovupd %ymm15, 32(%rdx)
        addq    $32, %rdx
        subq    $96, %rdi
        cmpq    $4, %r10
        jl      .Ltmp1
        addq    $32, %rdx
        addq    $64, %rsi
        subq    $32, %rdi
        cmpq    $4, %r11
        jl      .Ltmp0
        movq    %rbp, %rsp
        popq    %rbp

        #NO_APP
        popq    %r12
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        retq
.Lfunc_end0:
        .size   matmul_colmaj, .Lfunc_end0-matmul_colmaj
        .cfi_endproc
                                        # -- End function
        .globl  matmul                          # -- Begin function matmul
        .p2align        4
        .type   matmul,@function
matmul:                                 # @matmul
        .cfi_startproc
# %bb.0:
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        subq    $32, %rsp
        movq    %rdi, -8(%rbp)
        movq    %rsi, -16(%rbp)
        movq    %rdx, -24(%rbp)
        movq    -24(%rbp), %rdi
        movq    -16(%rbp), %rsi
        movq    -8(%rbp), %rdx
        callq   matmul_colmaj
        addq    $32, %rsp
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        retq
.Lfunc_end1:
        .size   matmul, .Lfunc_end1-matmul
        .cfi_endproc
                                        # -- End function
        .ident  "clang version 20.1.1 (https://github.com/llvm/llvm-project 424c2d9b7e4de40d0804dd374721e6411c27d1d1)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .addrsig_sym matmul_colmaj



# CHECK:       A

# CHECK-NEXT:  [0.00, 1.00, 5.00, 5.00],
# CHECK-NEXT:  [3.00, 4.00, 4.00, 4.00],
# CHECK-NEXT:  [4.00, 3.00, 6.00, 3.00],
# CHECK-NEXT:  [3.00, 0.00, 4.00, 2.00]

# CHECK-NEXT: B

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00],
# CHECK-NEXT:  [9.00, 4.00, 5.00, 3.00],
# CHECK-NEXT:  [2.00, 7.00, 0.00, 7.00],
# CHECK-NEXT:  [5.00, 5.00, 6.00, 2.00]

# CHECK-NEXT: C

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00],
# CHECK-NEXT:  [4.00, 0.00, 5.00, 0.00],
# CHECK-NEXT:  [6.00, 2.00, 4.00, 0.00],
# CHECK-NEXT:  [3.00, 7.00, 8.00, 1.00]

# CHECK-NEXT: A_colmaj

# CHECK-NEXT:  [0.00, 1.00, 5.00, 5.00],
# CHECK-NEXT:  [3.00, 4.00, 4.00, 4.00],
# CHECK-NEXT:  [4.00, 3.00, 6.00, 3.00],
# CHECK-NEXT:  [3.00, 0.00, 4.00, 2.00]

# CHECK-NEXT: B_colmaj

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00],
# CHECK-NEXT:  [9.00, 4.00, 5.00, 3.00],
# CHECK-NEXT:  [2.00, 7.00, 0.00, 7.00],
# CHECK-NEXT:  [5.00, 5.00, 6.00, 2.00]

# CHECK-NEXT: C_colmaj

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00],
# CHECK-NEXT:  [4.00, 0.00, 5.00, 0.00],
# CHECK-NEXT:  [6.00, 2.00, 4.00, 0.00],
# CHECK-NEXT:  [3.00, 7.00, 8.00, 1.00]

# CHECK-NEXT: C out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00],
# CHECK-NEXT:  [92.00, 73.00, 73.00, 66.00],
# CHECK-NEXT:  [92.00, 83.00, 69.00, 81.00],
# CHECK-NEXT:  [45.00, 54.00, 44.00, 51.00]

# CHECK-NEXT: C_asm out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00],
# CHECK-NEXT:  [92.00, 73.00, 73.00, 66.00],
# CHECK-NEXT:  [92.00, 83.00, 69.00, 81.00],
# CHECK-NEXT:  [45.00, 54.00, 44.00, 51.00]

# CHECK-EMPTY:
# CHECK-NEXT: Test Passed: The results are equal!
