module {
  transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
    transform.structured.vectorize %arg0 : !transform.any_op
    transform.yield
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @_vecto failures(suppress) (%tiled_linalg_op_0) : (!transform.any_op) -> ()
    %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loops_1 {factor=4} : !transform.any_op
    transform.loop.unroll %loops {factor=4} : !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.transfer_to_scf full_unroll = true
      transform.apply_patterns.vector.lower_transfer
    } : !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.memref.expand_strided_metadata
    } : !transform.any_op
    transform.yield
  }
}
