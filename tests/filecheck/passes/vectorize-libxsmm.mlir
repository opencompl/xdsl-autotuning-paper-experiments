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
// CHECK-NEXT:      %1 = arith.constant 16 : index
// CHECK-NEXT:      %2 = arith.constant 42 : index
// CHECK-NEXT:      %3 = arith.constant 16 : index
// CHECK-NEXT:      %4 = arith.constant 4 : index
// CHECK-NEXT:      %5 = ptr_xdsl.to_ptr %A : memref<3x42xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %6 = ptr_xdsl.to_ptr %B : memref<42x16xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %7 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:      %8 = arith.muli %7, %2 : index
// CHECK-NEXT:      %9 = arith.muli %7, %4 : index
// CHECK-NEXT:      %10 = arith.muli %7, %1 : index
// CHECK-NEXT:      %11 = arith.muli %7, %3 : index
// CHECK-NEXT:      %12 = arith.subi %11, %10 : index
// CHECK-NEXT:      %13 = ptr_xdsl.ptradd %5, %8 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %14 = ptr_xdsl.ptradd %13, %8 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %15 = vector.load %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %16 = vector.load %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %17 = vector.load %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %18 = vector.load %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %19 = vector.load %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %20 = vector.load %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %21 = vector.load %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %22 = vector.load %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %23 = vector.load %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %24 = vector.load %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %25 = vector.load %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %26 = vector.load %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42 = scf.for %43 = %c0 to %0 step %c1 iter_args(%44 = %5, %45 = %13, %46 = %14, %47 = %6, %48 = %15, %49 = %16, %50 = %17, %51 = %18, %52 = %19, %53 = %20, %54 = %21, %55 = %22, %56 = %23, %57 = %24, %58 = %25, %59 = %26) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>) {
// CHECK-NEXT:        %60 = ptr_xdsl.load %44 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %61 = ptr_xdsl.load %45 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %62 = ptr_xdsl.load %46 : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %63 = vector.broadcast %60 : f64 to vector<4xf64>
// CHECK-NEXT:        %64 = vector.broadcast %61 : f64 to vector<4xf64>
// CHECK-NEXT:        %65 = vector.broadcast %62 : f64 to vector<4xf64>
// CHECK-NEXT:        %66 = ptr_xdsl.load %47 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %67 = vector.fma %63, %66, %48 : vector<4xf64>
// CHECK-NEXT:        %68 = vector.fma %64, %66, %49 : vector<4xf64>
// CHECK-NEXT:        %69 = vector.fma %65, %66, %50 : vector<4xf64>
// CHECK-NEXT:        %70 = ptr_xdsl.ptradd %47, %9 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %71 = ptr_xdsl.load %70 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %72 = vector.fma %63, %71, %51 : vector<4xf64>
// CHECK-NEXT:        %73 = vector.fma %64, %71, %52 : vector<4xf64>
// CHECK-NEXT:        %74 = vector.fma %65, %71, %53 : vector<4xf64>
// CHECK-NEXT:        %75 = ptr_xdsl.ptradd %70, %9 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %76 = ptr_xdsl.load %75 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %77 = vector.fma %63, %76, %54 : vector<4xf64>
// CHECK-NEXT:        %78 = vector.fma %64, %76, %55 : vector<4xf64>
// CHECK-NEXT:        %79 = vector.fma %65, %76, %56 : vector<4xf64>
// CHECK-NEXT:        %80 = ptr_xdsl.ptradd %75, %9 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %81 = ptr_xdsl.load %80 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %82 = vector.fma %63, %81, %57 : vector<4xf64>
// CHECK-NEXT:        %83 = vector.fma %64, %81, %58 : vector<4xf64>
// CHECK-NEXT:        %84 = vector.fma %65, %81, %59 : vector<4xf64>
// CHECK-NEXT:        %85 = ptr_xdsl.ptradd %80, %9 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %86 = ptr_xdsl.ptradd %85, %12 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %87 = ptr_xdsl.ptradd %44, %7 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %88 = ptr_xdsl.ptradd %45, %7 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %89 = ptr_xdsl.ptradd %46, %7 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        scf.yield %87, %88, %89, %86, %67, %68, %69, %72, %73, %74, %77, %78, %79, %82, %83, %84 : !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      vector.store %31, %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %32, %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %33, %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %34, %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %35, %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %36, %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %37, %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %38, %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %39, %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %40, %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %41, %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %42, %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
