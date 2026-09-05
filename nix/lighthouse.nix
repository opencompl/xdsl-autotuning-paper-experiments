# The Lighthouse project (https://github.com/libxsmm/lighthouse), built from the
# `lighthouse` flake input.
#
# Building it here rather than installing it with `uv` keeps its MLIR bindings
# out of this repository's Python environment, and — more importantly — makes
# the upstream pipeline descriptors and transform schedules available as package
# data, so the evaluation runs lighthouse's own schedules instead of a copy.
{ lib
, buildPythonPackage
, setuptools
, mlir-python-bindings
, numpy
, pyyaml
, src
}:

buildPythonPackage {
  pname = "lighthouse";
  # Mirrors `lighthouse.__version__`; the flake input's revision is the real
  # pin and is recorded in flake.lock.
  version = "0.1.0a1";
  inherit src;

  pyproject = true;
  build-system = [ setuptools ];

  dependencies = [ mlir-python-bindings numpy pyyaml ];

  # lighthouse is pinned to the exact bindings nightly it was tested against.
  # Fail loudly here when the flake input moves past the version that
  # nix/mlir-python-bindings.nix packages, instead of building something that
  # only breaks halfway through a pipeline.
  postPatch = ''
    grep -q 'mlir-python-bindings==${mlir-python-bindings.version}' pyproject.toml || {
      echo "lighthouse pins a different mlir-python-bindings than nix/mlir-python-bindings.nix packages:" >&2
      grep 'mlir-python-bindings' pyproject.toml >&2
      exit 1
    }
  '';

  # The test suite is lit-based and needs the dev extras; the pipeline itself is
  # exercised by this repository's own `snakemake tests`.
  doCheck = false;

  pythonImportsCheck = [
    "lighthouse"
    "lighthouse.pipeline"
    "lighthouse.execution.runner"
  ];

  meta = {
    description = "MLIR Lighthouse: reference ingress, schedules and pipelines for MLIR";
    homepage = "https://github.com/libxsmm/lighthouse";
    license = with lib.licenses; [ asl20 llvm-exception ];
  };
}
