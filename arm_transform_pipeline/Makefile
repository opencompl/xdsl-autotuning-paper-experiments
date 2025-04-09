IMAGE_NAME = llvm-mlir

DOCKER_RUN = sudo docker run --rm -v $(PWD):/workspace -w /workspace $(IMAGE_NAME)

all: build analyze

build:
	sudo docker build -t $(IMAGE_NAME) ..

matmul-transform.mlir: matmul.mlir transform.mlir
	(echo "module attributes {transform.with_named_sequence} {" && cat $^ && echo "}") > $@

matmul.llvm.mlir: matmul-transform.mlir
	$(DOCKER_RUN) mlir-opt $< \
		--no-implicit-module \
		--transform-interpreter \
		--test-transform-dialect-erase-schedule \
		--canonicalize --cse --sccp \
		--expand-strided-metadata \
		--convert-linalg-to-loops \
		--lower-affine \
		--convert-vector-to-scf \
		--scf-forall-to-parallel \
		--convert-scf-to-openmp \
		--canonicalize --cse --sccp \
		--convert-scf-to-cf \
		--canonicalize --cse --sccp \
		--buffer-results-to-out-params \
		--convert-func-to-llvm=use-bare-ptr-memref-call-conv \
		--finalize-memref-to-llvm \
		--canonicalize --cse --sccp \
		--convert-vector-to-llvm=enable-x86vector \
		--convert-index-to-llvm \
		--convert-arith-to-llvm \
		--canonicalize --cse --sccp \
		--convert-cf-to-llvm \
		--convert-openmp-to-llvm \
		--canonicalize --cse --sccp \
		-o $@

matmul.ir: matmul.llvm.mlir
	$(DOCKER_RUN) mlir-translate --mlir-to-llvmir $< -o $@

matmul.bc: matmul.ir
	$(DOCKER_RUN) opt -O3 -march aarch64 --mcpu apple-m2 $< -o $@

matmul.s: matmul.bc
	$(DOCKER_RUN) llc -O3 -filetype=asm -march aarch64 --mcpu apple-m2 $< -o $@

analyze: matmul.s
	$(DOCKER_RUN) llvm-mca --march aarch64 --mcpu apple-m2 $< 2> /dev/null | grep "Total Cycles"

clean:
	rm -f matmul-transform.mlir matmul.llvm.mlir matmul.ir matmul.bc matmul.s
