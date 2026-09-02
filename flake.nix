{
  description = "xDSL small matrix experiments";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The Lighthouse project provides the `lighthouse` variant's pipeline
    # descriptors and transform schedules. Taking the repository as an input
    # means the evaluation runs the upstream schedules, pinned in flake.lock,
    # rather than a copy checked into this repository.
    lighthouse = {
      url = "github:libxsmm/lighthouse";
      flake = false;
    };
  };

  outputs = { nixpkgs, flake-utils, lighthouse, ... }:
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

          # Lighthouse and its MLIR bindings live in their own Python 3.12
          # environment: they carry a full nightly LLVM/MLIR of their own and
          # have no business in the uv environment the rest of the repository
          # uses. `scripts/lighthouse_codegen.py` runs under this interpreter.
          lighthousePython = pkgs.python312;
          mlirPythonBindings =
            lighthousePython.pkgs.callPackage ./nix/mlir-python-bindings.nix { };
          lighthousePackage = lighthousePython.pkgs.callPackage ./nix/lighthouse.nix {
            src = lighthouse;
            mlir-python-bindings = mlirPythonBindings;
          };
          lighthouseEnv = lighthousePython.withPackages (_: [ lighthousePackage ]);
          # Exposed under its own name so it does not shadow the `python` of
          # the uv environment, and wrapped so that codegen works from a plain
          # shell:
          #  * `lscpu` is how lighthouse enumerates host CPU features;
          #  * its pipelines parallelize with OpenMP, so the JIT that emits the
          #    object file has to load `libomp`. Lighthouse locates it with
          #    ctypes' `find_library`, which shells out to `$CC`/`ld` (hence
          #    LIBRARY_PATH and LD_LIBRARY_PATH), and then dlopens it through
          #    `libdl` to read back its path -- and glibc 2.34 merged libdl into
          #    libc, so a linkable `libdl.so` has to be put back on the path for
          #    that lookup to resolve at all.
          libdlCompat = pkgs.runCommand "libdl-compat" { } ''
            mkdir -p $out/lib
            for name in libdl.so libdl.so.2; do
              ln -s ${pkgs.glibc}/lib/libdl.so.2 $out/lib/$name
            done
          '';
          lighthouseInterpreter =
            let
              libDirs = [ "${pkgs.llvmPackages_22.openmp}/lib" ]
                ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux "${libdlCompat}/lib";
              wrapperArgs = [
                "--set-default CC ${pkgs.llvmPackages_22.clang}/bin/clang"
              ] ++ pkgs.lib.concatMap (dir: [
                "--prefix LIBRARY_PATH : ${dir}"
                "--prefix LD_LIBRARY_PATH : ${dir}"
              ]) libDirs ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux
                "--prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.util-linux pkgs.binutils ]}";
            in
            pkgs.runCommand "lighthouse-python"
              { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                makeWrapper ${lighthouseEnv}/bin/python $out/bin/lighthouse-python \
                  ${pkgs.lib.concatStringsSep " \\\n                  " wrapperArgs}
              '';
          # Only where the eudsl index publishes wheels we have hashes for.
          lighthouseSupported = builtins.elem system [ "x86_64-linux" "aarch64-darwin" ];
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
            ] ++ lib.optional lighthouseSupported lighthouseInterpreter
              ++ (if stdenv.hostPlatform.isLinux then [
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
            } // pkgs.lib.optionalAttrs lighthouseSupported {
              lighthouse = lighthousePackage;
              lighthouse-python = lighthouseInterpreter;
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
