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
          };
          llvmToolchain = with pkgs; buildEnv {
            name = "llvm-toolchain";
            paths = [
              uv
              llvmPackages_20.mlir
              llvmPackages_20.clang
              llvmPackages_20.lld
              llvmPackages_20.llvm.out
            ];
          };
        in
          {
            packages.default = llvmToolchain;

            devShells.default = with pkgs; mkShell {
              LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ];
              buildInputs = [
                llvmToolchain
                nodejs_22
              ];
            };
          }
    );
}
