# MLIR Python bindings, as consumed by the Lighthouse project.
#
# Lighthouse pins `mlir-python-bindings` to a nightly build published on the
# eudsl package index (https://llvm.github.io/eudsl), which is a plain list of
# wheels rather than a source release. `version` below must match the pin in
# lighthouse's `pyproject.toml`; bump both together when the lighthouse flake
# input moves, and refresh the hashes with
#
#   nix store prefetch-file --hash-type sha256 <wheel url>
{ lib
, stdenv
, buildPythonPackage
, fetchurl
, autoPatchelfHook
, python
, ml-dtypes
, numpy
, pyyaml
, libxml2
, ncurses
, zlib
}:

let
  version = "20260826+6c81b990d";

  # Only the platforms this repository's machines run on. Adding one is a
  # matter of prefetching the matching wheel from the index above.
  wheels = {
    x86_64-linux = {
      platform = "manylinux_2_34_x86_64";
      hash = "sha256-7bmCk7SySISXGDaGweq+nKYUw5WvD+IfnhH6HWQq+MI=";
    };
    aarch64-darwin = {
      platform = "macosx_13_0_arm64";
      hash = "sha256-h1wLRafTVxuj57JOEwqwJmUxIsrbWRaTOGTgt5jV434=";
    };
  };

  inherit (stdenv.hostPlatform) system;

  wheel = wheels.${system} or (throw
    "mlir-python-bindings: no wheel recorded for ${system}; add one from https://llvm.github.io/eudsl");

  pyTag = "cp${lib.replaceStrings [ "." ] [ "" ] python.pythonVersion}";

  fileName = "mlir_python_bindings-${version}-${pyTag}-${pyTag}-${wheel.platform}.whl";
in
buildPythonPackage {
  pname = "mlir-python-bindings";
  inherit version;
  format = "wheel";

  src = fetchurl {
    name = fileName;
    # `+` is not valid unescaped in a URL path.
    url = "https://github.com/llvm/eudsl/releases/download/mlir-python-bindings/"
      + lib.replaceStrings [ "+" ] [ "%2B" ] fileName;
    inherit (wheel) hash;
  };

  nativeBuildInputs = lib.optionals stdenv.hostPlatform.isLinux [ autoPatchelfHook ];

  buildInputs = lib.optionals stdenv.hostPlatform.isLinux [
    stdenv.cc.cc.lib
    libxml2
    ncurses
    zlib
  ];

  dependencies = [ ml-dtypes numpy pyyaml ];

  # The wheel caps PyYAML at 6.0.1 and ml_dtypes at 0.6.0; nixpkgs ships newer
  # ones and the bindings do not depend on anything that changed.
  pythonRelaxDeps = [ "PyYAML" "ml_dtypes" ];

  pythonImportsCheck = [ "mlir.ir" "mlir.execution_engine" ];

  meta = {
    description = "Upstream MLIR Python bindings, nightly wheels from the eudsl index";
    homepage = "https://github.com/llvm/eudsl";
    license = with lib.licenses; [ asl20 llvm-exception ];
    platforms = lib.attrNames wheels;
  };
}
