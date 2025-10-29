// RUN: libxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>) {
// CHECK-NEXT:    %{{.*}} = x86.ds.mov %{{.*}} : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.ds.mov %{{.*}} : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ds.mov %{{.*}} : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %{{.*}} = x86.get_register : () -> !x86.reg<rbp>
// CHECK-NEXT:    %{{.*}} = x86.get_register : () -> !x86.reg<rsp>
// CHECK-NEXT:    %{{.*}} = x86.s.push %{{.*}}, %{{.*}} : (!x86.reg<rsp>, !x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %{{.*}} = x86.ds.mov %{{.*}} : (!x86.reg<rsp>) -> !x86.reg<rbp>
// CHECK-NEXT:    %{{.*}} = x86.ri.sub %{{.*}}, 192 : (!x86.reg<rsp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %{{.*}} = x86.di.mov -64 : () -> !x86.reg<r10>
// CHECK-NEXT:    %{{.*}} = x86.rs.and %{{.*}}, %{{.*}} : (!x86.reg<rsp>, !x86.reg<r10>) -> !x86.reg<rsp>
// CHECK-NEXT:    %{{.*}} = x86.di.mov 0 : () -> !x86.reg<r11>
// CHECK-NEXT:    x86.c.jmp ^{{.*}}(%{{.*}} : !x86.reg<r11>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>):
// CHECK-NEXT:    x86.label "33"
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 3 : (!x86.reg<r11>) -> !x86.reg<r11>
// CHECK-NEXT:    %{{.*}} = x86.di.mov 0 : () -> !x86.reg<r10>
// CHECK-NEXT:    x86.c.jmp ^{{.*}}(%{{.*}} : !x86.reg<r10>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>):
// CHECK-NEXT:    x86.label "34"
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 16 : (!x86.reg<r10>) -> !x86.reg<r10>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 0 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 64 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 128 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 192 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 256 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 320 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %{{.*}} = x86.di.mov 0 : () -> !x86.reg<r12>
// CHECK-NEXT:    x86.c.jmp ^{{.*}}(%{{.*}} : !x86.reg<r12>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>, %{{.*}} : !x86.avx512reg<zmm26>, %{{.*}} : !x86.avx512reg<zmm27>, %{{.*}} : !x86.avx512reg<zmm28>, %{{.*}} : !x86.avx512reg<zmm29>, %{{.*}} : !x86.avx512reg<zmm30>, %{{.*}} : !x86.avx512reg<zmm31>, %{{.*}} : !x86.reg<r12>):
// CHECK-NEXT:    x86.label "35"
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 4 : (!x86.reg<r12>) -> !x86.reg<r12>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 0 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 64 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 0 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 64 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 0 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 64 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 0 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:    %{{.*}} = x86.dm.vmovapd %{{.*}}, 64 : (!x86.reg<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %{{.*}} = x86.dm.vbroadcastsd %{{.*}}, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %{{.*}} = x86.rss.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %{{.*}} = x86.si.cmp %{{.*}}, 64 : (!x86.reg<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %{{.*}} : !x86.rflags<rflags>, ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>, %{{.*}} : !x86.avx512reg<zmm26>, %{{.*}} : !x86.avx512reg<zmm27>, %{{.*}} : !x86.avx512reg<zmm28>, %{{.*}} : !x86.avx512reg<zmm29>, %{{.*}} : !x86.avx512reg<zmm30>, %{{.*}} : !x86.avx512reg<zmm31>, %{{.*}} : !x86.reg<r12>), ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>, %{{.*}} : !x86.avx512reg<zmm26>, %{{.*}} : !x86.avx512reg<zmm27>, %{{.*}} : !x86.avx512reg<zmm28>, %{{.*}} : !x86.avx512reg<zmm29>, %{{.*}} : !x86.avx512reg<zmm30>, %{{.*}} : !x86.avx512reg<zmm31>, %{{.*}} : !x86.reg<r12>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>, %{{.*}} : !x86.avx512reg<zmm26>, %{{.*}} : !x86.avx512reg<zmm27>, %{{.*}} : !x86.avx512reg<zmm28>, %{{.*}} : !x86.avx512reg<zmm29>, %{{.*}} : !x86.avx512reg<zmm30>, %{{.*}} : !x86.avx512reg<zmm31>, %{{.*}} : !x86.reg<r12>):
// CHECK-NEXT:    %{{.*}} = x86.ri.sub %{{.*}}, 512 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 128 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %{{.*}} = x86.ri.sub %{{.*}}, 8064 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.si.cmp %{{.*}}, 16 : (!x86.reg<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %{{.*}} : !x86.rflags<rflags>, ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>), ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>):
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 256 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %{{.*}} = x86.ri.add %{{.*}}, 1536 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %{{.*}} = x86.ri.sub %{{.*}}, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %{{.*}} = x86.si.cmp %{{.*}}, 3 : (!x86.reg<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %{{.*}} : !x86.rflags<rflags>, ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>), ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>)
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : !x86.reg<rdi>, %{{.*}} : !x86.reg<rsi>, %{{.*}} : !x86.reg<rdx>, %{{.*}} : !x86.reg<rbp>, %{{.*}} : !x86.reg<rsp>, %{{.*}} : !x86.reg<r10>, %{{.*}} : !x86.reg<r11>):
// CHECK-NEXT:    %{{.*}} = x86.ds.mov %{{.*}} : (!x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg<rsp>) -> (!x86.reg<rbp>, !x86.reg<rsp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
