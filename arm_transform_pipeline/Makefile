IMAGE_NAME = llvm-mlir
MLIR_INPUT = matmul-transform.mlir
MLIR_TRANSFORMED = matmul-transform.llvm.mlir
LLVM_IR = matmul-transform.ir
LLVM_BC = matmul-transform.bc
OBJ_FILE = matmul-transform.o

DOCKER_RUN = sudo docker run --rm -v $(PWD):/workspace -w /workspace $(IMAGE_NAME)

all: build run_opt translate optimize objdump

build:
	sudo docker build -t $(IMAGE_NAME) ..

run_opt: $(MLIR_INPUT)
	$(DOCKER_RUN) mlir-opt $< \
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
		-o $(MLIR_TRANSFORMED)

translate: $(MLIR_TRANSFORMED)
	$(DOCKER_RUN) mlir-translate --mlir-to-llvmir $< -o $(LLVM_IR)

optimize: $(LLVM_IR)
	$(DOCKER_RUN) opt -O3 -march aarch64 --mcpu apple-m2 $< -o $(LLVM_BC)
	$(DOCKER_RUN) llc -O3 -filetype=obj -march aarch64 --mcpu apple-m2 $< -o $(OBJ_FILE)

objdump: $(OBJ_FILE)
	$(DOCKER_RUN) aarch64-linux-gnu-objdump --disassemble $(OBJ_FILE)

clean:
	rm -f $(MLIR_TRANSFORMED) $(LLVM_IR) $(LLVM_BC) $(OBJ_FILE)

.PHONY: all build run_opt translate optimize objdump clean
