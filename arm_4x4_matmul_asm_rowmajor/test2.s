	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	_matrix_mul_4x4_asm             ; -- Begin function matrix_mul_4x4_asm
	.p2align	2
_matrix_mul_4x4_asm:                    ; @matrix_mul_4x4_asm
	.cfi_startproc
; %bb.0:
	ldr	x8, [sp, #56]
	ldr	x9, [sp]
	ldp	q2, q3, [x1]
	ldp	q0, q4, [x9]
	ldp	q1, q6, [x8]
	ldp	q5, q17, [x9, #32]
	ldp	q7, q16, [x1, #32]
	fmla.4s	v2, v1, v0[0]
	fmla.4s	v3, v1, v4[0]
	fmla.4s	v7, v1, v5[0]
	fmla.4s	v16, v1, v17[0]
	fmla.4s	v2, v6, v0[1]
	fmla.4s	v3, v6, v4[1]
	fmla.4s	v7, v6, v5[1]
	fmla.4s	v16, v6, v17[1]
	ldp	q1, q6, [x8, #32]
	fmla.4s	v2, v1, v0[2]
	fmla.4s	v3, v1, v4[2]
	fmla.4s	v7, v1, v5[2]
	fmla.4s	v16, v1, v17[2]
	fmla.4s	v2, v6, v0[3]
	fmla.4s	v3, v6, v4[3]
	fmla.4s	v7, v6, v5[3]
	fmla.4s	v16, v6, v17[3]
	stp	q2, q3, [x1]
	stp	q7, q16, [x1, #32]
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
