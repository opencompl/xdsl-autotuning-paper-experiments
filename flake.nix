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
          # Pinned libxsmm revision rather than the 1.17 release in nixpkgs.
          libxsmmPinned = pkgs.libxsmm.overrideAttrs (old: {
            version = "1.17-unstable-10b7dc82";
            src = pkgs.fetchFromGitHub {
              owner = "libxsmm";
              repo = "libxsmm";
              rev = "10b7dc82b3c46157e76eb40e4e959555f895b24d";
              hash = "sha256-iEltpVqRgbMNbNQryJ/wI0OSNtgnuSuu6bl0TfVCLB4=";
            };
            patches = [ ./nix/libxsmm-rpath.patch ];
            # documentation/{LICENSE,CONTRIBUTING}.md are symlinks into the
            # source root, so `make install` copies them as links that dangle
            # once the docs land in their own output.
            postInstall = old.postInstall + ''
              cp --remove-destination LICENSE.md CONTRIBUTING.md \
                ''${!outputDoc}/share/libxsmm/
            '';
          });
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
              libxsmmPinned
              papi
            ] ++ lib.optionals stdenv.hostPlatform.isx86_64 [
              aoclBlas
            ] else [ ]);
          };
        in
          {
            packages = {
              default = llvmToolchain;
              libxsmm = libxsmmPinned;
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
