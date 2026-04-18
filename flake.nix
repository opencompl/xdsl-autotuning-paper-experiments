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
            paths = [
              uv
              pkg-config
              llvmPackages_20.mlir
              llvmPackages_20.clang
              llvmPackages_20.lld
              llvmPackages_20.llvm.out
            ] ++ (if stdenv.hostPlatform.isLinux then [
              mkl
            ] else [
              llvmPackages_20.openmp
            ]);
          };
        in
          {
            packages.default = llvmToolchain;

            devShells.default = with pkgs; mkShellNoCC {
              LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ];
              nativeBuildInputs = [ pkg-config ];
              buildInputs = [
                llvmToolchain
                nodejs_22
              ];
            };
          }
    );
}
