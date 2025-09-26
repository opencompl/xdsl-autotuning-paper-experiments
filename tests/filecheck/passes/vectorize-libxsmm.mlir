// RUN: xdsl-opt -p vectorize-libxsmm %s | filecheck %s

func.func @matmul(
  %A: memref<3x42xf64>,
  %B: memref<42x16xf64>,
  %C: memref<3x16xf64>
) {
  linalg.matmul ins(%A, %B: memref<3x42xf64>, memref<42x16xf64>) outs(%C: memref<3x16xf64>)
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @matmul(%A : memref<3x42xf64>, %B : memref<42x16xf64>, %C : memref<3x16xf64>) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %c3 = arith.constant 3 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c5 = arith.constant 5 : index
// CHECK-NEXT:      %c6 = arith.constant 6 : index
// CHECK-NEXT:      %c7 = arith.constant 7 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c9 = arith.constant 9 : index
// CHECK-NEXT:      %c10 = arith.constant 10 : index
// CHECK-NEXT:      %c11 = arith.constant 11 : index
// CHECK-NEXT:      %c12 = arith.constant 12 : index
// CHECK-NEXT:      %c13 = arith.constant 13 : index
// CHECK-NEXT:      %c14 = arith.constant 14 : index
// CHECK-NEXT:      %c15 = arith.constant 15 : index
// CHECK-NEXT:      %c16 = arith.constant 16 : index
// CHECK-NEXT:      %0 = arith.constant 42 : index
// CHECK-NEXT:      %1 = arith.constant 4 : index
// CHECK-NEXT:      %2 = ptr_xdsl.to_ptr %A : memref<3x42xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %3 = ptr_xdsl.to_ptr %B : memref<42x16xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %4 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:      %5 = arith.muli %4, %0 : index
// CHECK-NEXT:      %6 = arith.muli %4, %1 : index
// CHECK-NEXT:      %7 = ptr_xdsl.ptradd %2, %5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %8 = ptr_xdsl.ptradd %7, %5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %9 = vector.load %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %10 = vector.load %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %11 = vector.load %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %12 = vector.load %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %13 = vector.load %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %14 = vector.load %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %15 = vector.load %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %16 = vector.load %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %17 = vector.load %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %18 = vector.load %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %19 = vector.load %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %20 = vector.load %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36 = scf.for %37 = %c0 to %0 step %c1 iter_args(%38 = %2, %39 = %7, %40 = %8, %41 = %3, %42 = %9, %43 = %10, %44 = %11, %45 = %12, %46 = %13, %47 = %14, %48 = %15, %49 = %16, %50 = %17, %51 = %18, %52 = %19, %53 = %20) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>) {
// CHECK-NEXT:        %54 = ptr_xdsl.load %38 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %55 = ptr_xdsl.load %39 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %56 = ptr_xdsl.load %40 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %57 = vector.broadcast %54 : f64 to vector<4xf64>
// CHECK-NEXT:        %58 = vector.broadcast %55 : f64 to vector<4xf64>
// CHECK-NEXT:        %59 = vector.broadcast %56 : f64 to vector<4xf64>
// CHECK-NEXT:        %60 = ptr_xdsl.load %41 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %61 = vector.fma %57, %60, %42 : vector<4xf64>
// CHECK-NEXT:        %62 = vector.fma %58, %60, %43 : vector<4xf64>
// CHECK-NEXT:        %63 = vector.fma %59, %60, %44 : vector<4xf64>
// CHECK-NEXT:        %64 = ptr_xdsl.ptradd %41, %6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %65 = ptr_xdsl.load %64 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %66 = vector.fma %57, %65, %45 : vector<4xf64>
// CHECK-NEXT:        %67 = vector.fma %58, %65, %46 : vector<4xf64>
// CHECK-NEXT:        %68 = vector.fma %59, %65, %47 : vector<4xf64>
// CHECK-NEXT:        %69 = ptr_xdsl.ptradd %64, %6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %70 = ptr_xdsl.load %69 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %71 = vector.fma %57, %70, %48 : vector<4xf64>
// CHECK-NEXT:        %72 = vector.fma %58, %70, %49 : vector<4xf64>
// CHECK-NEXT:        %73 = vector.fma %59, %70, %50 : vector<4xf64>
// CHECK-NEXT:        %74 = ptr_xdsl.ptradd %69, %6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %75 = ptr_xdsl.load %74 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %76 = vector.fma %57, %75, %51 : vector<4xf64>
// CHECK-NEXT:        %77 = vector.fma %58, %75, %52 : vector<4xf64>
// CHECK-NEXT:        %78 = vector.fma %59, %75, %53 : vector<4xf64>
// CHECK-NEXT:        %79 = ptr_xdsl.ptradd %74, %6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %80 = ptr_xdsl.ptradd %38, %4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %81 = ptr_xdsl.ptradd %39, %4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %82 = ptr_xdsl.ptradd %40, %4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        scf.yield %80, %81, %82, %79, %61, %62, %63, %66, %67, %68, %71, %72, %73, %76, %77, %78 : !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      vector.store %25, %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %26, %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %27, %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %28, %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %29, %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %30, %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %31, %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %32, %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %33, %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %34, %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %35, %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %36, %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
