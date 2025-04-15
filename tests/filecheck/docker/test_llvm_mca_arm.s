// RUN: llvm-mca --march aarch64 --mcpu apple-m2 %s | filecheck %s

myfun:
	ldp	q0, q1, [x0]
	ldp	q2, q3, [x0, #32]
	ldp	q4, q5, [x1]
	ldp	q6, q7, [x1, #32]
	ldp	q16, q17, [x2]
	ldp	q18, q19, [x2, #32]
	fmla	v16.4s, v4.4s, v0.s[0]
	fmla	v17.4s, v4.4s, v1.s[0]
	fmla	v18.4s, v4.4s, v2.s[0]
	fmla	v19.4s, v4.4s, v3.s[0]
	fmla	v16.4s, v5.4s, v0.s[1]
	fmla	v17.4s, v5.4s, v1.s[1]
	fmla	v18.4s, v5.4s, v2.s[1]
	fmla	v19.4s, v5.4s, v3.s[1]
	fmla	v16.4s, v6.4s, v0.s[2]
	fmla	v17.4s, v6.4s, v1.s[2]
	fmla	v18.4s, v6.4s, v2.s[2]
	fmla	v19.4s, v6.4s, v3.s[2]
	fmla	v16.4s, v7.4s, v0.s[3]
	fmla	v17.4s, v7.4s, v1.s[3]
	fmla	v18.4s, v7.4s, v2.s[3]
	fmla	v19.4s, v7.4s, v3.s[3]
	stp	q16, q17, [x2]
	stp	q18, q19, [x2, #32]
	ret        
// CHECK:       Iterations:        100
// CHECK-NEXT:  Instructions:      2500
// CHECK-NEXT:  Total Cycles:      587
// CHECK-NEXT:  Total uOps:        3300
// CHECK-NEXT:  
// CHECK-NEXT:  Dispatch Width:    6
// CHECK-NEXT:  uOps Per Cycle:    5.62
// CHECK-NEXT:  IPC:               4.26
// CHECK-NEXT:  Block RThroughput: 5.5
// CHECK-NEXT:  
// CHECK-NEXT:  
// CHECK-NEXT:  Instruction Info:
// CHECK-NEXT:  [1]: #uOps
// CHECK-NEXT:  [2]: Latency
// CHECK-NEXT:  [3]: RThroughput
// CHECK-NEXT:  [4]: MayLoad
// CHECK-NEXT:  [5]: MayStore
// CHECK-NEXT:  [6]: HasSideEffects (U)
// CHECK-NEXT:  
// CHECK-NEXT:  [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
// CHECK-NEXT:   2      4     0.50    *                   ldp	q0, q1, [x0]
// CHECK-NEXT:   2      4     0.50    *                   ldp	q2, q3, [x0, #32]
// CHECK-NEXT:   2      4     0.50    *                   ldp	q4, q5, [x1]
// CHECK-NEXT:   2      4     0.50    *                   ldp	q6, q7, [x1, #32]
// CHECK-NEXT:   2      4     0.50    *                   ldp	q16, q17, [x2]
// CHECK-NEXT:   2      4     0.50    *                   ldp	q18, q19, [x2, #32]
// CHECK-NEXT:   1      2     0.33                        fmla	v16.4s, v4.4s, v0.s[0]
// CHECK-NEXT:   1      2     0.33                        fmla	v17.4s, v4.4s, v1.s[0]
// CHECK-NEXT:   1      2     0.33                        fmla	v18.4s, v4.4s, v2.s[0]
// CHECK-NEXT:   1      2     0.33                        fmla	v19.4s, v4.4s, v3.s[0]
// CHECK-NEXT:   1      2     0.33                        fmla	v16.4s, v5.4s, v0.s[1]
// CHECK-NEXT:   1      2     0.33                        fmla	v17.4s, v5.4s, v1.s[1]
// CHECK-NEXT:   1      2     0.33                        fmla	v18.4s, v5.4s, v2.s[1]
// CHECK-NEXT:   1      2     0.33                        fmla	v19.4s, v5.4s, v3.s[1]
// CHECK-NEXT:   1      2     0.33                        fmla	v16.4s, v6.4s, v0.s[2]
// CHECK-NEXT:   1      2     0.33                        fmla	v17.4s, v6.4s, v1.s[2]
// CHECK-NEXT:   1      2     0.33                        fmla	v18.4s, v6.4s, v2.s[2]
// CHECK-NEXT:   1      2     0.33                        fmla	v19.4s, v6.4s, v3.s[2]
// CHECK-NEXT:   1      2     0.33                        fmla	v16.4s, v7.4s, v0.s[3]
// CHECK-NEXT:   1      2     0.33                        fmla	v17.4s, v7.4s, v1.s[3]
// CHECK-NEXT:   1      2     0.33                        fmla	v18.4s, v7.4s, v2.s[3]
// CHECK-NEXT:   1      2     0.33                        fmla	v19.4s, v7.4s, v3.s[3]
// CHECK-NEXT:   2      4     1.00           *            stp	q16, q17, [x2]
// CHECK-NEXT:   2      4     1.00           *            stp	q18, q19, [x2, #32]
// CHECK-NEXT:   1      0     1.00                  U     ret
// CHECK-NEXT:  
// CHECK-NEXT:  
// CHECK-NEXT:  Resources:
// CHECK-NEXT:  [0.0] - CyUnitB
// CHECK-NEXT:  [0.1] - CyUnitB
// CHECK-NEXT:  [1]   - CyUnitBR
// CHECK-NEXT:  [2.0] - CyUnitFloatDiv
// CHECK-NEXT:  [2.1] - CyUnitFloatDiv
// CHECK-NEXT:  [3.0] - CyUnitI
// CHECK-NEXT:  [3.1] - CyUnitI
// CHECK-NEXT:  [3.2] - CyUnitI
// CHECK-NEXT:  [3.3] - CyUnitI
// CHECK-NEXT:  [4]   - CyUnitID
// CHECK-NEXT:  [5]   - CyUnitIM
// CHECK-NEXT:  [6.0] - CyUnitIS
// CHECK-NEXT:  [6.1] - CyUnitIS
// CHECK-NEXT:  [7]   - CyUnitIntDiv
// CHECK-NEXT:  [8.0] - CyUnitLS
// CHECK-NEXT:  [8.1] - CyUnitLS
// CHECK-NEXT:  [9.0] - CyUnitV
// CHECK-NEXT:  [9.1] - CyUnitV
// CHECK-NEXT:  [9.2] - CyUnitV
// CHECK-NEXT:  [10]  - CyUnitVC
// CHECK-NEXT:  [11]  - CyUnitVD
// CHECK-NEXT:  [12.0] - CyUnitVM
// CHECK-NEXT:  [12.1] - CyUnitVM
// CHECK-NEXT:  
// CHECK-NEXT:  
// CHECK-NEXT:  Resource pressure per iteration:
// CHECK-NEXT:  [0.0]  [0.1]  [1]    [2.0]  [2.1]  [3.0]  [3.1]  [3.2]  [3.3]  [4]    [5]    [6.0]  [6.1]  [7]    [8.0]  [8.1]  [9.0]  [9.1]  [9.2]  [10]   [11]   [12.0] [12.1] 
// CHECK-NEXT:  0.50   0.50   1.00    -      -     0.25   0.25   0.25   0.25    -      -      -      -      -     4.99   5.01   5.33   5.33   5.34    -      -      -      -     
// CHECK-NEXT:  
// CHECK-NEXT:  Resource pressure by instruction:
// CHECK-NEXT:  [0.0]  [0.1]  [1]    [2.0]  [2.1]  [3.0]  [3.1]  [3.2]  [3.3]  [4]    [5]    [6.0]  [6.1]  [7]    [8.0]  [8.1]  [9.0]  [9.1]  [9.2]  [10]   [11]   [12.0] [12.1] Instructions:
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.61   0.39    -      -      -      -      -      -      -     ldp	q0, q1, [x0]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.39   0.61    -      -      -      -      -      -      -     ldp	q2, q3, [x0, #32]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.62   0.38    -      -      -      -      -      -      -     ldp	q4, q5, [x1]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.37   0.63    -      -      -      -      -      -      -     ldp	q6, q7, [x1, #32]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.63   0.37    -      -      -      -      -      -      -     ldp	q16, q17, [x2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.37   0.63    -      -      -      -      -      -      -     ldp	q18, q19, [x2, #32]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.61   0.32   0.07    -      -      -      -     fmla	v16.4s, v4.4s, v0.s[0]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.60   0.07   0.33    -      -      -      -     fmla	v17.4s, v4.4s, v1.s[0]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.61   0.33   0.06    -      -      -      -     fmla	v18.4s, v4.4s, v2.s[0]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.62   0.05   0.33    -      -      -      -     fmla	v19.4s, v4.4s, v3.s[0]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.33   0.34   0.33    -      -      -      -     fmla	v16.4s, v5.4s, v0.s[1]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.34   0.33   0.33    -      -      -      -     fmla	v17.4s, v5.4s, v1.s[1]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.33   0.33   0.34    -      -      -      -     fmla	v18.4s, v5.4s, v2.s[1]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.33   0.34   0.33    -      -      -      -     fmla	v19.4s, v5.4s, v3.s[1]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.05   0.34   0.61    -      -      -      -     fmla	v16.4s, v6.4s, v0.s[2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.35   0.34   0.31    -      -      -      -     fmla	v17.4s, v6.4s, v1.s[2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.06   0.32   0.62    -      -      -      -     fmla	v18.4s, v6.4s, v2.s[2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.32   0.61   0.07    -      -      -      -     fmla	v19.4s, v6.4s, v3.s[2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.05   0.35   0.60    -      -      -      -     fmla	v16.4s, v7.4s, v0.s[3]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.35   0.34   0.31    -      -      -      -     fmla	v17.4s, v7.4s, v1.s[3]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.07   0.31   0.62    -      -      -      -     fmla	v18.4s, v7.4s, v2.s[3]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.31   0.61   0.08    -      -      -      -     fmla	v19.4s, v7.4s, v3.s[3]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     0.80   1.20    -      -      -      -      -      -      -     stp	q16, q17, [x2]
// CHECK-NEXT:   -      -      -      -      -      -      -      -      -      -      -      -      -      -     1.20   0.80    -      -      -      -      -      -      -     stp	q18, q19, [x2, #32]
// CHECK-NEXT:  0.50   0.50   1.00    -      -     0.25   0.25   0.25   0.25    -      -      -      -      -      -      -      -      -      -      -      -      -      -     ret
