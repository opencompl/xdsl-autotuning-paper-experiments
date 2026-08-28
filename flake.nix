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
          clangUnwrapped = pkgs.writeShellScriptBin "clang-unwrapped" ''
            exec ${pkgs.llvmPackages_22.clang-unwrapped}/bin/clang "$@"
          '';
          aoclBlas = pkgs.callPackage ./nix/aocl-blas.nix { };
          llvmToolchain = with pkgs; buildEnv {
            name = "llvm-toolchain";
            ignoreCollisions = true;
            paths = [
              uv
              pkg-config
              clangUnwrapped
              llvmPackages_22.mlir
              llvmPackages_22.clang
              llvmPackages_22.lld
              llvmPackages_22.llvm.out
              llvmPackages_22.openmp
            ] ++ (if stdenv.hostPlatform.isLinux then [
              mkl
              libxsmm
              papi
            ] ++ lib.optionals stdenv.hostPlatform.isx86_64 [
              aoclBlas
            ] else [ ]);
          };
        in
          {
            packages = {
              default = llvmToolchain;
            } // pkgs.lib.optionalAttrs (
              pkgs.stdenv.hostPlatform.isLinux
              && pkgs.stdenv.hostPlatform.isx86_64
            ) {
              aocl-blas = aoclBlas;
            };

            devShells.default = with pkgs; mkShellNoCC {
              # XTC shells out to mlir-opt/mlir-translate/opt/llc; it resolves
              # them from these prefixes (expecting {prefix}/bin/...) ahead of
              # its own mlir/llvm wheels, whose binaries abort on some hosts.
              XTC_MLIR_PREFIX = "${llvmToolchain}";
              XTC_LLVM_PREFIX = "${llvmToolchain}";
              LD_LIBRARY_PATH = lib.makeLibraryPath ([ stdenv.cc.cc.lib zlib llvmToolchain ]
                ++ lib.optionals stdenv.hostPlatform.isLinux [ papi ]);
              LIBRARY_PATH = lib.makeLibraryPath [ llvmToolchain ];
              C_INCLUDE_PATH = "${llvmToolchain}/include";
              nativeBuildInputs = [ pkg-config ];
              buildInputs = [
                llvmToolchain
                nodejs_22
              ];
            };
          }
    );
}
