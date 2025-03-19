module {
  llvm.func @matrix_mul_4x4_asm(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.undef : vector<4xf32>
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.mul %10, %9 : i64
    %12 = llvm.add %11, %10 : i64
    %13 = llvm.getelementptr %arg1[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %15 = llvm.mul %8, %9 : i64
    %16 = llvm.add %15, %10 : i64
    %17 = llvm.getelementptr %arg1[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %19 = llvm.mul %7, %9 : i64
    %20 = llvm.add %19, %10 : i64
    %21 = llvm.getelementptr %arg1[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %22 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %23 = llvm.mul %6, %9 : i64
    %24 = llvm.add %23, %10 : i64
    %25 = llvm.getelementptr %arg1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.load %25 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %27 = llvm.getelementptr %arg2[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.load %27 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %29 = llvm.getelementptr %arg2[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %30 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %31 = llvm.getelementptr %arg2[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %33 = llvm.getelementptr %arg2[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %34 = llvm.load %33 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %35 = llvm.getelementptr %arg0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %36 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %37 = llvm.getelementptr %arg0[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %39 = llvm.getelementptr %arg0[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %41 = llvm.getelementptr %arg0[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %42 = llvm.load %41 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>
    %43 = llvm.extractelement %14[%5 : i64] : vector<4xf32>
    %44 = llvm.insertelement %43, %4[%3 : i32] : vector<4xf32>
    %45 = llvm.shufflevector %44, %4 [0, 0, 0, 0] : vector<4xf32> 
    %46 = llvm.intr.fmuladd(%28, %45, %36)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %47 = llvm.extractelement %14[%2 : i64] : vector<4xf32>
    %48 = llvm.insertelement %47, %4[%3 : i32] : vector<4xf32>
    %49 = llvm.shufflevector %48, %4 [0, 0, 0, 0] : vector<4xf32> 
    %50 = llvm.intr.fmuladd(%30, %49, %46)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %51 = llvm.extractelement %14[%1 : i64] : vector<4xf32>
    %52 = llvm.insertelement %51, %4[%3 : i32] : vector<4xf32>
    %53 = llvm.shufflevector %52, %4 [0, 0, 0, 0] : vector<4xf32> 
    %54 = llvm.intr.fmuladd(%32, %53, %50)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %55 = llvm.extractelement %14[%0 : i64] : vector<4xf32>
    %56 = llvm.insertelement %55, %4[%3 : i32] : vector<4xf32>
    %57 = llvm.shufflevector %56, %4 [0, 0, 0, 0] : vector<4xf32> 
    %58 = llvm.intr.fmuladd(%34, %57, %54)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %59 = llvm.extractelement %18[%5 : i64] : vector<4xf32>
    %60 = llvm.insertelement %59, %4[%3 : i32] : vector<4xf32>
    %61 = llvm.shufflevector %60, %4 [0, 0, 0, 0] : vector<4xf32> 
    %62 = llvm.intr.fmuladd(%28, %61, %38)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %63 = llvm.extractelement %18[%2 : i64] : vector<4xf32>
    %64 = llvm.insertelement %63, %4[%3 : i32] : vector<4xf32>
    %65 = llvm.shufflevector %64, %4 [0, 0, 0, 0] : vector<4xf32> 
    %66 = llvm.intr.fmuladd(%30, %65, %62)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %67 = llvm.extractelement %18[%1 : i64] : vector<4xf32>
    %68 = llvm.insertelement %67, %4[%3 : i32] : vector<4xf32>
    %69 = llvm.shufflevector %68, %4 [0, 0, 0, 0] : vector<4xf32> 
    %70 = llvm.intr.fmuladd(%32, %69, %66)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %71 = llvm.extractelement %18[%0 : i64] : vector<4xf32>
    %72 = llvm.insertelement %71, %4[%3 : i32] : vector<4xf32>
    %73 = llvm.shufflevector %72, %4 [0, 0, 0, 0] : vector<4xf32> 
    %74 = llvm.intr.fmuladd(%34, %73, %70)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %75 = llvm.extractelement %22[%5 : i64] : vector<4xf32>
    %76 = llvm.insertelement %75, %4[%3 : i32] : vector<4xf32>
    %77 = llvm.shufflevector %76, %4 [0, 0, 0, 0] : vector<4xf32> 
    %78 = llvm.intr.fmuladd(%28, %77, %40)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %79 = llvm.extractelement %22[%2 : i64] : vector<4xf32>
    %80 = llvm.insertelement %79, %4[%3 : i32] : vector<4xf32>
    %81 = llvm.shufflevector %80, %4 [0, 0, 0, 0] : vector<4xf32> 
    %82 = llvm.intr.fmuladd(%30, %81, %78)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %83 = llvm.extractelement %22[%1 : i64] : vector<4xf32>
    %84 = llvm.insertelement %83, %4[%3 : i32] : vector<4xf32>
    %85 = llvm.shufflevector %84, %4 [0, 0, 0, 0] : vector<4xf32> 
    %86 = llvm.intr.fmuladd(%32, %85, %82)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %87 = llvm.extractelement %22[%0 : i64] : vector<4xf32>
    %88 = llvm.insertelement %87, %4[%3 : i32] : vector<4xf32>
    %89 = llvm.shufflevector %88, %4 [0, 0, 0, 0] : vector<4xf32> 
    %90 = llvm.intr.fmuladd(%34, %89, %86)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %91 = llvm.extractelement %26[%5 : i64] : vector<4xf32>
    %92 = llvm.insertelement %91, %4[%3 : i32] : vector<4xf32>
    %93 = llvm.shufflevector %92, %4 [0, 0, 0, 0] : vector<4xf32> 
    %94 = llvm.intr.fmuladd(%28, %93, %42)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %95 = llvm.extractelement %26[%2 : i64] : vector<4xf32>
    %96 = llvm.insertelement %95, %4[%3 : i32] : vector<4xf32>
    %97 = llvm.shufflevector %96, %4 [0, 0, 0, 0] : vector<4xf32> 
    %98 = llvm.intr.fmuladd(%30, %97, %94)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %99 = llvm.extractelement %26[%1 : i64] : vector<4xf32>
    %100 = llvm.insertelement %99, %4[%3 : i32] : vector<4xf32>
    %101 = llvm.shufflevector %100, %4 [0, 0, 0, 0] : vector<4xf32> 
    %102 = llvm.intr.fmuladd(%32, %101, %98)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %103 = llvm.extractelement %26[%0 : i64] : vector<4xf32>
    %104 = llvm.insertelement %103, %4[%3 : i32] : vector<4xf32>
    %105 = llvm.shufflevector %104, %4 [0, 0, 0, 0] : vector<4xf32> 
    %106 = llvm.intr.fmuladd(%34, %105, %102)  : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    llvm.store %58, %35 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr
    llvm.store %74, %37 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr
    llvm.store %90, %39 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr
    llvm.store %106, %41 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr
    llvm.return
  }
}
