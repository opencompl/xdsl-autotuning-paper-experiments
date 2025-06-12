// RUN: clang -c %s -o %t
// RUN: /opt/uica-staticdeps/uiCA.py -arch SKL %t -TPonly | filecheck %s

myfun:
	vbroadcastss (%rdi),%xmm4
	vmovups (%rsi),%xmm3
	vmovups 0x10(%rsi),%xmm2
	vmovups 0x20(%rsi),%xmm1
	vmovups 0x30(%rsi),%xmm0
	vfmadd213ps (%rdx),%xmm3,%xmm4
	vbroadcastss 0x4(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm2,%xmm5
	vbroadcastss 0x8(%rdi),%xmm4
	vfmadd213ps %xmm5,%xmm1,%xmm4
	vbroadcastss 0xc(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm0,%xmm5
	vmovups %xmm5,(%rdx)
	vbroadcastss 0x10(%rdi),%xmm4
	vfmadd213ps 0x10(%rdx),%xmm3,%xmm4
	vbroadcastss 0x14(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm2,%xmm5
	vbroadcastss 0x18(%rdi),%xmm4
	vfmadd213ps %xmm5,%xmm1,%xmm4
	vbroadcastss 0x1c(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm0,%xmm5
	vmovups %xmm5,0x10(%rdx)
	vbroadcastss 0x20(%rdi),%xmm4
	vfmadd213ps 0x20(%rdx),%xmm3,%xmm4
	vbroadcastss 0x24(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm2,%xmm5
	vbroadcastss 0x28(%rdi),%xmm4
	vfmadd213ps %xmm5,%xmm1,%xmm4
	vbroadcastss 0x2c(%rdi),%xmm5
	vfmadd213ps %xmm4,%xmm0,%xmm5
	vmovups %xmm5,0x20(%rdx)
	vbroadcastss 0x30(%rdi),%xmm4
	vfmadd213ps 0x30(%rdx),%xmm3,%xmm4
	vbroadcastss 0x34(%rdi),%xmm3
	vfmadd213ps %xmm4,%xmm2,%xmm3
	vbroadcastss 0x38(%rdi),%xmm2
	vfmadd213ps %xmm3,%xmm1,%xmm2
	vbroadcastss 0x3c(%rdi),%xmm1
	vfmadd213ps %xmm2,%xmm0,%xmm1
	vmovups %xmm1,0x30(%rdx)
	ret

// CHECK: 20.00
