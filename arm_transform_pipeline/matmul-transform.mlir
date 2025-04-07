module attributes {transform.with_named_sequence} {
  func.func @myfun(%arg0: memref<4x4xf32> {llvm.noalias}, %arg1: memref<4x4xf32> {llvm.noalias}, %arg2: memref<4x4xf32> {llvm.noalias}) {
    linalg.matmul {__id0__} ins(%arg0, %arg1 : memref<4x4xf32>, memref<4x4xf32>) outs(%arg2 : memref<4x4xf32>)
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "__id0__i" : !transform.any_op
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "__id0__k" : !transform.any_op
    %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %2 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_contraction
    } : !transform.any_op
    transform.yield 
  }
}

