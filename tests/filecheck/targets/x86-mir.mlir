// RUN: xdsl-opt %s -t x86-mir | filecheck %s

// The compxsmm-llvm-gemm-passes pipeline printed as LLVM Machine IR: a K loop
// whose carried values become PHIs, and a masked tail along M.

// CHECK:      ---
// CHECK-NEXT: name: matmul
// CHECK-NEXT: tracksRegLiveness: true
// CHECK-NEXT: registers:
// CHECK-NEXT:   - { id: 0, class: gr64 }
// CHECK-NEXT:   - { id: 1, class: gr64 }
// CHECK-NEXT:   - { id: 2, class: gr64 }
// CHECK-NEXT:   - { id: 3, class: gr64 }
// CHECK-NEXT:   - { id: 4, class: gr64 }
// CHECK-NEXT:   - { id: 5, class: gr64 }
// CHECK-NEXT:   - { id: 6, class: vr512 }
// CHECK-NEXT:   - { id: 7, class: vr512 }
// CHECK-NEXT:   - { id: 8, class: vr512 }
// CHECK-NEXT:   - { id: 9, class: vr512 }
// CHECK-NEXT:   - { id: 10, class: gr64 }
// CHECK-NEXT:   - { id: 11, class: gr64 }
// CHECK-NEXT:   - { id: 12, class: gr64 }
// CHECK-NEXT:   - { id: 13, class: vr512 }
// CHECK-NEXT:   - { id: 14, class: vr512 }
// CHECK-NEXT:   - { id: 15, class: vr512 }
// CHECK-NEXT:   - { id: 16, class: vr512 }
// CHECK-NEXT:   - { id: 17, class: gr64 }
// CHECK-NEXT:   - { id: 18, class: gr32 }
// CHECK-NEXT:   - { id: 19, class: vk8wm }
// CHECK-NEXT:   - { id: 20, class: vr512 }
// CHECK-NEXT:   - { id: 21, class: vr512 }
// CHECK-NEXT:   - { id: 22, class: vr512 }
// CHECK-NEXT:   - { id: 23, class: vr512 }
// CHECK-NEXT:   - { id: 24, class: gr64 }
// CHECK-NEXT:   - { id: 25, class: gr64 }
// CHECK-NEXT:   - { id: 26, class: vr512 }
// CHECK-NEXT:   - { id: 27, class: vr512 }
// CHECK-NEXT:   - { id: 28, class: vr512 }
// CHECK-NEXT:   - { id: 29, class: vr512 }
// CHECK-NEXT:   - { id: 30, class: vr512 }
// CHECK-NEXT:   - { id: 31, class: vr512 }
// CHECK-NEXT:   - { id: 32, class: gr64 }
// CHECK-NEXT:   - { id: 33, class: gr64 }
// CHECK-NEXT:   - { id: 34, class: vr512 }
// CHECK-NEXT:   - { id: 35, class: vr512 }
// CHECK-NEXT:   - { id: 36, class: vr512 }
// CHECK-NEXT:   - { id: 37, class: vr512 }
// CHECK-NEXT:   - { id: 38, class: vr512 }
// CHECK-NEXT:   - { id: 39, class: vr512 }
// CHECK-NEXT:   - { id: 40, class: vr512 }
// CHECK-NEXT:   - { id: 41, class: vr512 }
// CHECK-NEXT:   - { id: 42, class: gr64 }
// CHECK-NEXT:   - { id: 43, class: gr64 }
// CHECK-NEXT:   - { id: 44, class: vr512 }
// CHECK-NEXT:   - { id: 45, class: vr512 }
// CHECK-NEXT:   - { id: 46, class: vr512 }
// CHECK-NEXT:   - { id: 47, class: vr512 }
// CHECK-NEXT:   - { id: 48, class: vr512 }
// CHECK-NEXT:   - { id: 49, class: vr512 }
// CHECK-NEXT:   - { id: 50, class: vr512 }
// CHECK-NEXT:   - { id: 51, class: vr512 }
// CHECK-NEXT:   - { id: 52, class: gr64 }
// CHECK-NEXT:   - { id: 53, class: gr64 }
// CHECK-NEXT:   - { id: 54, class: vr512 }
// CHECK-NEXT:   - { id: 55, class: vr512 }
// CHECK-NEXT:   - { id: 56, class: vr512 }
// CHECK-NEXT:   - { id: 57, class: vr512 }
// CHECK-NEXT:   - { id: 58, class: vr512 }
// CHECK-NEXT:   - { id: 59, class: vr512 }
// CHECK-NEXT:   - { id: 60, class: vr512 }
// CHECK-NEXT:   - { id: 61, class: vr512 }
// CHECK-NEXT:   - { id: 62, class: gr64 }
// CHECK-NEXT:   - { id: 63, class: gr64 }
// CHECK-NEXT:   - { id: 64, class: vr512 }
// CHECK-NEXT:   - { id: 65, class: vr512 }
// CHECK-NEXT:   - { id: 66, class: gr64 }
// CHECK-NEXT:   - { id: 67, class: gr64 }
// CHECK-NEXT:   - { id: 68, class: gr64 }
// CHECK-NEXT:   - { id: 69, class: gr64 }
// CHECK-NEXT:   - { id: 70, class: gr64 }
// CHECK-NEXT:   - { id: 71, class: gr64 }
// CHECK-NEXT: liveins:
// CHECK-NEXT:   - { reg: '$rdi' }
// CHECK-NEXT:   - { reg: '$rsi' }
// CHECK-NEXT:   - { reg: '$rdx' }
// CHECK-NEXT: body: |
// CHECK-NEXT:   bb.0:
// CHECK-NEXT:     successors: %bb.1
// CHECK-NEXT:     liveins: $rdi, $rsi, $rdx
// CHECK-NEXT:     %0 = COPY $rdi
// CHECK-NEXT:     %1 = COPY $rsi
// CHECK-NEXT:     %2 = COPY $rdx
// CHECK-NEXT:     %17 = MOV64ri32 3
// CHECK-NEXT:     %18 = COPY %17.sub_32bit
// CHECK-NEXT:     %19 = KMOVBkr %18
// CHECK-NEXT:     %20 = VMOVUPDZrm %2, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %21 = VMOVUPDZrmkz %19, %2, 1, $noreg, 64, $noreg
// CHECK-NEXT:     %22 = VMOVUPDZrm %2, 1, $noreg, 80, $noreg
// CHECK-NEXT:     %23 = VMOVUPDZrmkz %19, %2, 1, $noreg, 144, $noreg
// CHECK-NEXT:     %24 = MOV64ri32 0
// CHECK-EMPTY:
// CHECK-NEXT:   bb.1:
// CHECK-NEXT:     successors: %bb.1, %bb.2
// CHECK-NEXT:     %3 = PHI %24, %bb.0, %25, %bb.1
// CHECK-NEXT:     %4 = PHI %0, %bb.0, %63, %bb.1
// CHECK-NEXT:     %5 = PHI %1, %bb.0, %62, %bb.1
// CHECK-NEXT:     %6 = PHI %20, %bb.0, %59, %bb.1
// CHECK-NEXT:     %7 = PHI %21, %bb.0, %60, %bb.1
// CHECK-NEXT:     %8 = PHI %22, %bb.0, %64, %bb.1
// CHECK-NEXT:     %9 = PHI %23, %bb.0, %65, %bb.1
// CHECK-NEXT:     %25 = ADD64ri32 %3, 4, implicit-def dead $eflags
// CHECK-NEXT:     %26 = VMOVUPDZrm %4, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %27 = VMOVUPDZrmkz %19, %4, 1, $noreg, 64, $noreg
// CHECK-NEXT:     %28 = VBROADCASTSDZrm %5, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %29 = VFMADD231PDZr %6, %26, %28, implicit $mxcsr
// CHECK-NEXT:     %30 = VFMADD231PDZr %7, %27, %28, implicit $mxcsr
// CHECK-NEXT:     %31 = VBROADCASTSDZrm %5, 1, $noreg, 512, $noreg
// CHECK-NEXT:     %32 = ADD64ri32 %5, 8, implicit-def dead $eflags
// CHECK-NEXT:     %33 = ADD64ri32 %4, 80, implicit-def dead $eflags
// CHECK-NEXT:     %34 = VFMADD231PDZr %8, %26, %31, implicit $mxcsr
// CHECK-NEXT:     %35 = VFMADD231PDZr %9, %27, %31, implicit $mxcsr
// CHECK-NEXT:     %36 = VMOVUPDZrm %33, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %37 = VMOVUPDZrmkz %19, %33, 1, $noreg, 64, $noreg
// CHECK-NEXT:     %38 = VBROADCASTSDZrm %32, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %39 = VFMADD231PDZr %29, %36, %38, implicit $mxcsr
// CHECK-NEXT:     %40 = VFMADD231PDZr %30, %37, %38, implicit $mxcsr
// CHECK-NEXT:     %41 = VBROADCASTSDZrm %32, 1, $noreg, 512, $noreg
// CHECK-NEXT:     %42 = ADD64ri32 %32, 8, implicit-def dead $eflags
// CHECK-NEXT:     %43 = ADD64ri32 %33, 80, implicit-def dead $eflags
// CHECK-NEXT:     %44 = VFMADD231PDZr %34, %36, %41, implicit $mxcsr
// CHECK-NEXT:     %45 = VFMADD231PDZr %35, %37, %41, implicit $mxcsr
// CHECK-NEXT:     %46 = VMOVUPDZrm %43, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %47 = VMOVUPDZrmkz %19, %43, 1, $noreg, 64, $noreg
// CHECK-NEXT:     %48 = VBROADCASTSDZrm %42, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %49 = VFMADD231PDZr %39, %46, %48, implicit $mxcsr
// CHECK-NEXT:     %50 = VFMADD231PDZr %40, %47, %48, implicit $mxcsr
// CHECK-NEXT:     %51 = VBROADCASTSDZrm %42, 1, $noreg, 512, $noreg
// CHECK-NEXT:     %52 = ADD64ri32 %42, 8, implicit-def dead $eflags
// CHECK-NEXT:     %53 = ADD64ri32 %43, 80, implicit-def dead $eflags
// CHECK-NEXT:     %54 = VFMADD231PDZr %44, %46, %51, implicit $mxcsr
// CHECK-NEXT:     %55 = VFMADD231PDZr %45, %47, %51, implicit $mxcsr
// CHECK-NEXT:     %56 = VMOVUPDZrm %53, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %57 = VMOVUPDZrmkz %19, %53, 1, $noreg, 64, $noreg
// CHECK-NEXT:     %58 = VBROADCASTSDZrm %52, 1, $noreg, 0, $noreg
// CHECK-NEXT:     %59 = VFMADD231PDZr %49, %56, %58, implicit $mxcsr
// CHECK-NEXT:     %60 = VFMADD231PDZr %50, %57, %58, implicit $mxcsr
// CHECK-NEXT:     %61 = VBROADCASTSDZrm %52, 1, $noreg, 512, $noreg
// CHECK-NEXT:     %62 = ADD64ri32 %52, 8, implicit-def dead $eflags
// CHECK-NEXT:     %63 = ADD64ri32 %53, 80, implicit-def dead $eflags
// CHECK-NEXT:     %64 = VFMADD231PDZr %54, %56, %61, implicit $mxcsr
// CHECK-NEXT:     %65 = VFMADD231PDZr %55, %57, %61, implicit $mxcsr
// CHECK-NEXT:     CMP64ri32 %25, 64, implicit-def $eflags
// CHECK-NEXT:     JCC_1 %bb.1, 12, implicit $eflags
// CHECK-NEXT:     JMP_1 %bb.2
// CHECK-EMPTY:
// CHECK-NEXT:   bb.2:
// CHECK-NEXT:     %10 = PHI %25, %bb.1
// CHECK-NEXT:     %11 = PHI %63, %bb.1
// CHECK-NEXT:     %12 = PHI %62, %bb.1
// CHECK-NEXT:     %13 = PHI %59, %bb.1
// CHECK-NEXT:     %14 = PHI %60, %bb.1
// CHECK-NEXT:     %15 = PHI %64, %bb.1
// CHECK-NEXT:     %16 = PHI %65, %bb.1
// CHECK-NEXT:     VMOVUPDZmr %2, 1, $noreg, 0, $noreg, %13
// CHECK-NEXT:     VMOVUPDZmrk %2, 1, $noreg, 64, $noreg, %19, %14
// CHECK-NEXT:     VMOVUPDZmr %2, 1, $noreg, 80, $noreg, %15
// CHECK-NEXT:     VMOVUPDZmrk %2, 1, $noreg, 144, $noreg, %19, %16
// CHECK-NEXT:     %66 = SUB64ri32 %11, 5040, implicit-def dead $eflags
// CHECK-NEXT:     %67 = SUB64ri32 %12, 512, implicit-def dead $eflags
// CHECK-NEXT:     %68 = ADD64ri32 %2, 80, implicit-def dead $eflags
// CHECK-NEXT:     %69 = SUB64ri32 %66, 80, implicit-def dead $eflags
// CHECK-NEXT:     %70 = ADD64ri32 %67, 1024, implicit-def dead $eflags
// CHECK-NEXT:     %71 = ADD64ri32 %68, 80, implicit-def dead $eflags
// CHECK-NEXT:     RET64
// CHECK-EMPTY:
// CHECK-NEXT: ...

x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
  %3 = x86.di.mov 3 : () -> !x86.reg64
  %4 = x86.ks.kmovb %3 : (!x86.reg64) -> !x86.avx512maskreg<k1>
  %5 = x86.dm.vmovupd [%2] : (!x86.reg64<rdx>) -> !x86.avx512reg
  %6 = x86.dmk.vmovupd[%2 + 64], %4 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %7 = x86.dm.vmovupd [%2 + 80] : (!x86.reg64<rdx>) -> !x86.avx512reg
  %8 = x86.dmk.vmovupd[%2 + 144], %4 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %9 = x86.di.mov 0 : () -> !x86.reg64
  x86.fallthrough ^bb1(%9 : !x86.reg64, %0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %5 : !x86.avx512reg, %6 : !x86.avx512reg, %7 : !x86.avx512reg, %8 : !x86.avx512reg)
^bb1(%10: !x86.reg64, %11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.avx512reg, %14: !x86.avx512reg, %15: !x86.avx512reg, %16: !x86.avx512reg):
  x86.label "scf_body_0_for"
  %17 = x86.ri.add %10, 4 : (!x86.reg64) -> !x86.reg64
  %18 = x86.dm.vmovupd [%11] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %19 = x86.dmk.vmovupd[%11 + 64], %4 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %20 = x86.dm.vbroadcastsd [%12] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %21 = x86.rss.vfmadd231pd %13, %18, %20 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %22 = x86.rss.vfmadd231pd %14, %19, %20 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %23 = x86.dm.vbroadcastsd [%12 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %24 = x86.ri.add %12, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %25 = x86.ri.add %11, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %26 = x86.rss.vfmadd231pd %15, %18, %23 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %27 = x86.rss.vfmadd231pd %16, %19, %23 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %28 = x86.dm.vmovupd [%25] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %29 = x86.dmk.vmovupd[%25 + 64], %4 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %30 = x86.dm.vbroadcastsd [%24] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %31 = x86.rss.vfmadd231pd %21, %28, %30 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %32 = x86.rss.vfmadd231pd %22, %29, %30 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %33 = x86.dm.vbroadcastsd [%24 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %34 = x86.ri.add %24, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %35 = x86.ri.add %25, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %36 = x86.rss.vfmadd231pd %26, %28, %33 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %37 = x86.rss.vfmadd231pd %27, %29, %33 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %38 = x86.dm.vmovupd [%35] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %39 = x86.dmk.vmovupd[%35 + 64], %4 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %40 = x86.dm.vbroadcastsd [%34] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %41 = x86.rss.vfmadd231pd %31, %38, %40 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %42 = x86.rss.vfmadd231pd %32, %39, %40 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %43 = x86.dm.vbroadcastsd [%34 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %44 = x86.ri.add %34, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %45 = x86.ri.add %35, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %46 = x86.rss.vfmadd231pd %36, %38, %43 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %47 = x86.rss.vfmadd231pd %37, %39, %43 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %48 = x86.dm.vmovupd [%45] : (!x86.reg64<rdi>) -> !x86.avx512reg
  %49 = x86.dmk.vmovupd[%45 + 64], %4 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg
  %50 = x86.dm.vbroadcastsd [%44] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %51 = x86.rss.vfmadd231pd %41, %48, %50 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %52 = x86.rss.vfmadd231pd %42, %49, %50 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %53 = x86.dm.vbroadcastsd [%44 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg
  %54 = x86.ri.add %44, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %55 = x86.ri.add %45, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %56 = x86.rss.vfmadd231pd %46, %48, %53 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %57 = x86.rss.vfmadd231pd %47, %49, %53 : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
  %58 = x86.si.cmp %17, 64 : (!x86.reg64) -> !x86.rflags<rflags>
  x86.c.jl %58 : !x86.rflags<rflags>, ^bb1(%17 : !x86.reg64, %55 : !x86.reg64<rdi>, %54 : !x86.reg64<rsi>, %51 : !x86.avx512reg, %52 : !x86.avx512reg, %56 : !x86.avx512reg, %57 : !x86.avx512reg), ^bb2(%17 : !x86.reg64, %55 : !x86.reg64<rdi>, %54 : !x86.reg64<rsi>, %51 : !x86.avx512reg, %52 : !x86.avx512reg, %56 : !x86.avx512reg, %57 : !x86.avx512reg)
^bb2(%59: !x86.reg64, %60: !x86.reg64<rdi>, %61: !x86.reg64<rsi>, %62: !x86.avx512reg, %63: !x86.avx512reg, %64: !x86.avx512reg, %65: !x86.avx512reg):
  x86.ms.vmovupd [%2], %62 : (!x86.reg64<rdx>, !x86.avx512reg) -> ()
  x86.msk.vmovupd[%2 + 64], %63, %4 : (!x86.reg64<rdx>, !x86.avx512reg, !x86.avx512maskreg<k1>) -> ()
  x86.ms.vmovupd [%2 + 80], %64 : (!x86.reg64<rdx>, !x86.avx512reg) -> ()
  x86.msk.vmovupd[%2 + 144], %65, %4 : (!x86.reg64<rdx>, !x86.avx512reg, !x86.avx512maskreg<k1>) -> ()
  %66 = x86.ri.sub %60, 5040 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %67 = x86.ri.sub %61, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %68 = x86.ri.add %2, 80 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
  %69 = x86.ri.sub %66, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
  %70 = x86.ri.add %67, 1024 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
  %71 = x86.ri.add %68, 80 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
  x86_func.ret
}
