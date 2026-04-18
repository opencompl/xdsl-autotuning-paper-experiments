{
  description = "xDSL small matrix experiments";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          llvmToolchain = with pkgs; buildEnv {
            name = "llvm-toolchain";
            ignoreCollisions = true;
            paths = [
              uv
              pkg-config
              llvmPackages_20.mlir
              llvmPackages_20.clang
              llvmPackages_20.lld
              llvmPackages_20.llvm.out
              llvmPackages_20.openmp
            ] ++ (if stdenv.hostPlatform.isLinux then [
              mkl
              libxsmm
            ] else [
            ]);
          };
        in
          {
            packages.default = llvmToolchain;

            devShells.default = with pkgs; mkShellNoCC {
              LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib zlib llvmToolchain ];
              nativeBuildInputs = [ pkg-config ];
              buildInputs = [
                llvmToolchain
                nodejs_22
              ];
            };
          }
    );
}
