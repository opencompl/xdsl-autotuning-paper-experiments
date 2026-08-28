{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  python3,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "aocl-blas";
  version = "5.3.2";

  src = fetchFromGitHub {
    owner = "amd";
    repo = "blis";
    rev = finalAttrs.version;
    hash = "sha256-Fa7YEOUJOxICoKKwCBo4jbvmEvWmQhg7PcwCNYXsBDM=";
  };

  nativeBuildInputs = [
    cmake
    python3
  ];

  cmakeFlags = [
    # Include all AMD Zen kernels and select between them at runtime. In
    # particular, this avoids specializing the package for the Nix builder.
    (lib.cmakeFeature "BLIS_CONFIG_FAMILY" "amdzen")
    (lib.cmakeBool "ENABLE_CBLAS" true)
    (lib.cmakeFeature "ENABLE_THREADING" "no")
    (lib.cmakeBool "BUILD_SHARED_LIBS" true)
    (lib.cmakeBool "BUILD_STATIC_LIBS" false)
  ];

  meta = {
    description = "AMD Optimizing CPU Libraries implementation of BLAS";
    homepage = "https://github.com/amd/blis";
    license = with lib.licenses; [
      bsd3
      mit
    ];
    platforms = [ "x86_64-linux" ];
  };
})
