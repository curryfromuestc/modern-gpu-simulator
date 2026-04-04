ifeq ($(GPUAPPS_SETUP_ENVIRONMENT_WAS_RUN), 0)
$(error You must run "source setup_environment before calling make")
endif

GENCODE_SM10 := -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 := -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 := -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 := -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM72 := -gencode=arch=compute_72,code=\"sm_72,compute_72\"
GENCODE_SM75 := -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM86 := -gencode=arch=compute_86,code=\"sm_86,compute_86\"
GENCODE_SM87 := -gencode=arch=compute_87,code=\"sm_87,compute_87\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
GENCODE_SM120 := -gencode=arch=compute_120,code=\"sm_120,compute_120\"

cuda-11: cuda-10
	echo "Currently no uBench designed for 11"

cuda-10: cuda-9
	echo "Currently no uBench designed for 10"

cuda-9: cuda-8
	echo "Currently no uBench designed for 9"

cuda-8: cuda-7
	echo "Currently no uBench designed for 7"

cuda-7: cuda-6
	echo "Currently no uBench designed for 7"

cuda-6: cuda-5
	echo "Currently no uBench designed for 6"

cuda-5: cuda-4 $(BINDIR)/$(BINSUBDIR)/microbench-oo-vfunc-test
	

cuda-4:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	echo "Currently no uBench designed for 4"

$(BINDIR)/$(BINSUBDIR)/microbench-oo-vfunc-test: oo-vfunc-test.cu
	nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM87) $(GENCODE_SM89) $(GENCODE_SM90) $(GENCODE_SM120) $(NVCC_ADDITIONAL_ARGS) oo-vfunc-test.cu -o $(BINDIR)/$(BINSUBDIR)/microbench-oo-vfunc-test
