
NVCC=nvcc

CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
CUDNN_PATH?=$(CUDA_LIB64)
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
BIN_DIR?=bin
MKDIR=mkdir -p
#BLAS
BLAS_LIBRARY?=cublas_static
BLAS_PATH?=$(CUDA_LIB64)
#CONV
NVRTC_STATIC_LIBS=-lnvrtc_static -lnvptxcompiler_static -lnvrtc-builtins_static
CONV_LIBRARY?=$(CUDNN_PATH)/libcudnn_cnn_static.a $(CUDNN_PATH)/libcudnn_adv_static.a $(CUDNN_PATH)/libcudnn_ops_static.a $(CUDNN_PATH)/libcudnn_heuristic_static.a $(CUDNN_PATH)/libcudnn_graph_static.a $(CUDNN_PATH)/libcudnn_engines_precompiled_static.a  $(CUDNN_PATH)/libcudnn_engines_runtime_compiled_static.a $(NVRTC_STATIC_LIBS)
CONV_PATH?=$
KERNELS_DIR=../kernels/
PAD_KERNELS?=1
COMMA=,

GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM72 ?= -gencode=arch=compute_72,code=\"sm_72,compute_72\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM80 ?= -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"
GENCODE_SM89 ?= -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 ?= -gencode=arch=compute_90,code=\"sm_90,compute_90\"
GENCODE_SM120 ?= -gencode=arch=compute_120,code=\"sm_120,compute_120\"

NVCC_ARCH_ARGS=$(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM87) $(GENCODE_SM89) $(GENCODE_SM90) $(GENCODE_SM120) -g -Xptxas -v

NVCC_ADDITIONAL_ARGS=-lineinfo
.PHONY=all gemm conv rnn all_reduce nccl_single nccl_mpi clean

all: gemm conv rnn

gemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) gemm_bench.cu -DUSE_TENSOR_CORES=0 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/gemm_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -ldl -lpthread -L $(BLAS_PATH) -lcublasLt_static -lz -lculibos -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand_static $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) gemm_bench.cu -DUSE_TENSOR_CORES=1 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/gemm_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -ldl -lpthread -L $(BLAS_PATH) -lcublasLt_static -lz -lculibos -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand_static $(CUDA_CPPFLAGS) 

conv:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) conv_bench.cu -DUSE_TENSOR_CORES=0 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/conv_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -ldl -lpthread -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcublasLt_static -lz -lcublas_static -lculibos -lcurand_static $(CONV_LIBRARY) $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) conv_bench.cu -DUSE_TENSOR_CORES=1 -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/conv_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -ldl -lpthread -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcublasLt_static -lz -lcublas_static -lculibos -lcurand_static $(CONV_LIBRARY) $(CUDA_CPPFLAGS) 

rnn:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) rnn_bench.cu -DUSE_TENSOR_CORES=0 -o $(BIN_DIR)/rnn_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -ldl -lpthread -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcublasLt_static -lcublas_static -lculibos -lz -lcurand_static $(CONV_LIBRARY) $(CUDA_CPPFLAGS) -std=c++14
	$(CUDA_PATH)/bin/$(NVCC) $(NVCC_ARCH_ARGS) $(NVCC_ADDITIONAL_ARGS) rnn_bench.cu -DUSE_TENSOR_CORES=1 -o $(BIN_DIR)/rnn_bench-tencore -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -ldl -lpthread -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcublasLt_static -lcublas_static -lz -lculibos -lcurand_static $(CONV_LIBRARY) $(CUDA_CPPFLAGS) 

all_reduce: nccl_single nccl_mpi

nccl_single:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_single_all_reduce.cu -o $(BIN_DIR)/nccl_single_all_reduce -I $(KERNELS_DIR) -I $(NCCL_PATH)/include/ -I $(CUDNN_PATH)/include/ -L $(NCCL_PATH)/lib/ -L $(CUDNN_PATH)/lib64 -lnccl -lcudart -lcurand_static $(CUDA_CPPFLAGS) 

nccl_mpi:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) nccl_mpi_all_reduce.cu -o $(BIN_DIR)/nccl_mpi_all_reduce -I $(KERNELS_DIR) -I $(NCCL_PATH)/include/ -I $(CUDNN_PATH)/include/ -I $(MPI_INCLUDE_PATH) -L $(NCCL_PATH)/lib/ -L $(CUDNN_PATH)/lib64 -L $(MPI_PATH)/lib -lnccl -lcurand_static -lcudart -lmpi $(CUDA_CPPFLAGS) 

sparse:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) sparse_bench.cu -o $(BIN_DIR)/sparse_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand_static -lcusparse $(CUDA_CPPFLAGS) 


clean:
	rm -rf $(BIN_DIR)

rebuild: clean all