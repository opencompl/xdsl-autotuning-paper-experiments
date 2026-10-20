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
          libxsmmPinned = pkgs.libxsmm.overrideAttrs (_: {
            version = "2.1.0-unstable-2026-09-28";
            src = pkgs.fetchFromGitHub {
              owner = "libxsmm";
              repo = "libxsmm";
              rev = "3057b239c16994d307589e40e45f40857ad5f5b8";
              hash = "sha256-eNzc9aoSwCjNTvVHRDcpeccEY9nJ6Jwz/iXxRHZ+Gko=";
            };
            patches = [ ./nix/libxsmm-rpath.patch ];
            # Upstream now installs .pc files under lib/pkgconfig, not lib.
            # documentation/{LICENSE,CONTRIBUTING}.md are symlinks into the
            # source root, so `make install` copies them as links that dangle
            # once the docs land in their own output.
            postInstall = ''
              moveToOutput "lib/pkgconfig" "$dev"
              moveToOutput "share/libxsmm" "$doc"
              cp --remove-destination LICENSE.md CONTRIBUTING.md \
                "$doc/share/libxsmm/"
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
              libxsmmPinned.dev
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
