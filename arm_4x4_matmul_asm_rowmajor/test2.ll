; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @matrix_mul_4x4_asm(ptr %0, ptr %1, ptr %2) {
  %4 = getelementptr float, ptr %1, i64 0
  %5 = load <4 x float>, ptr %4, align 4
  %6 = getelementptr float, ptr %1, i64 4
  %7 = load <4 x float>, ptr %6, align 4
  %8 = getelementptr float, ptr %1, i64 8
  %9 = load <4 x float>, ptr %8, align 4
  %10 = getelementptr float, ptr %1, i64 12
  %11 = load <4 x float>, ptr %10, align 4
  %12 = getelementptr float, ptr %2, i64 0
  %13 = load <4 x float>, ptr %12, align 4
  %14 = getelementptr float, ptr %2, i64 4
  %15 = load <4 x float>, ptr %14, align 4
  %16 = getelementptr float, ptr %2, i64 8
  %17 = load <4 x float>, ptr %16, align 4
  %18 = getelementptr float, ptr %2, i64 12
  %19 = load <4 x float>, ptr %18, align 4
  %20 = getelementptr float, ptr %0, i64 0
  %21 = load <4 x float>, ptr %20, align 4
  %22 = getelementptr float, ptr %0, i64 4
  %23 = load <4 x float>, ptr %22, align 4
  %24 = getelementptr float, ptr %0, i64 8
  %25 = load <4 x float>, ptr %24, align 4
  %26 = getelementptr float, ptr %0, i64 12
  %27 = load <4 x float>, ptr %26, align 4
  %28 = extractelement <4 x float> %5, i64 0
  %29 = insertelement <4 x float> undef, float %28, i32 0
  %30 = shufflevector <4 x float> %29, <4 x float> undef, <4 x i32> zeroinitializer
  %31 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %13, <4 x float> %30, <4 x float> %21)
  %32 = extractelement <4 x float> %5, i64 1
  %33 = insertelement <4 x float> undef, float %32, i32 0
  %34 = shufflevector <4 x float> %33, <4 x float> undef, <4 x i32> zeroinitializer
  %35 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> %34, <4 x float> %31)
  %36 = extractelement <4 x float> %5, i64 2
  %37 = insertelement <4 x float> undef, float %36, i32 0
  %38 = shufflevector <4 x float> %37, <4 x float> undef, <4 x i32> zeroinitializer
  %39 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %17, <4 x float> %38, <4 x float> %35)
  %40 = extractelement <4 x float> %5, i64 3
  %41 = insertelement <4 x float> undef, float %40, i32 0
  %42 = shufflevector <4 x float> %41, <4 x float> undef, <4 x i32> zeroinitializer
  %43 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %19, <4 x float> %42, <4 x float> %39)
  %44 = extractelement <4 x float> %7, i64 0
  %45 = insertelement <4 x float> undef, float %44, i32 0
  %46 = shufflevector <4 x float> %45, <4 x float> undef, <4 x i32> zeroinitializer
  %47 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %13, <4 x float> %46, <4 x float> %23)
  %48 = extractelement <4 x float> %7, i64 1
  %49 = insertelement <4 x float> undef, float %48, i32 0
  %50 = shufflevector <4 x float> %49, <4 x float> undef, <4 x i32> zeroinitializer
  %51 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> %50, <4 x float> %47)
  %52 = extractelement <4 x float> %7, i64 2
  %53 = insertelement <4 x float> undef, float %52, i32 0
  %54 = shufflevector <4 x float> %53, <4 x float> undef, <4 x i32> zeroinitializer
  %55 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %17, <4 x float> %54, <4 x float> %51)
  %56 = extractelement <4 x float> %7, i64 3
  %57 = insertelement <4 x float> undef, float %56, i32 0
  %58 = shufflevector <4 x float> %57, <4 x float> undef, <4 x i32> zeroinitializer
  %59 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %19, <4 x float> %58, <4 x float> %55)
  %60 = extractelement <4 x float> %9, i64 0
  %61 = insertelement <4 x float> undef, float %60, i32 0
  %62 = shufflevector <4 x float> %61, <4 x float> undef, <4 x i32> zeroinitializer
  %63 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %13, <4 x float> %62, <4 x float> %25)
  %64 = extractelement <4 x float> %9, i64 1
  %65 = insertelement <4 x float> undef, float %64, i32 0
  %66 = shufflevector <4 x float> %65, <4 x float> undef, <4 x i32> zeroinitializer
  %67 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> %66, <4 x float> %63)
  %68 = extractelement <4 x float> %9, i64 2
  %69 = insertelement <4 x float> undef, float %68, i32 0
  %70 = shufflevector <4 x float> %69, <4 x float> undef, <4 x i32> zeroinitializer
  %71 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %17, <4 x float> %70, <4 x float> %67)
  %72 = extractelement <4 x float> %9, i64 3
  %73 = insertelement <4 x float> undef, float %72, i32 0
  %74 = shufflevector <4 x float> %73, <4 x float> undef, <4 x i32> zeroinitializer
  %75 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %19, <4 x float> %74, <4 x float> %71)
  %76 = extractelement <4 x float> %11, i64 0
  %77 = insertelement <4 x float> undef, float %76, i32 0
  %78 = shufflevector <4 x float> %77, <4 x float> undef, <4 x i32> zeroinitializer
  %79 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %13, <4 x float> %78, <4 x float> %27)
  %80 = extractelement <4 x float> %11, i64 1
  %81 = insertelement <4 x float> undef, float %80, i32 0
  %82 = shufflevector <4 x float> %81, <4 x float> undef, <4 x i32> zeroinitializer
  %83 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> %82, <4 x float> %79)
  %84 = extractelement <4 x float> %11, i64 2
  %85 = insertelement <4 x float> undef, float %84, i32 0
  %86 = shufflevector <4 x float> %85, <4 x float> undef, <4 x i32> zeroinitializer
  %87 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %17, <4 x float> %86, <4 x float> %83)
  %88 = extractelement <4 x float> %11, i64 3
  %89 = insertelement <4 x float> undef, float %88, i32 0
  %90 = shufflevector <4 x float> %89, <4 x float> undef, <4 x i32> zeroinitializer
  %91 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %19, <4 x float> %90, <4 x float> %87)
  store <4 x float> %43, ptr %20, align 4
  store <4 x float> %59, ptr %22, align 4
  store <4 x float> %75, ptr %24, align 4
  store <4 x float> %91, ptr %26, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
