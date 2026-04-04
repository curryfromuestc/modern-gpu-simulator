GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM72 := -gencode=arch=compute_72,code=\"sm_72,compute_72\"
GENCODE_SM75 := -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM86 := -gencode=arch=compute_86,code=\"sm_86,compute_86\"
GENCODE_SM87 := -gencode=arch=compute_87,code=\"sm_87,compute_87\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
GENCODE_SM120 := -gencode=arch=compute_120,code=\"sm_120,compute_120\"

CUOPTS = $(GENCODE_SM50) $(GENCODE_SM61) $(GENCODE_SM70) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM87) $(GENCODE_SM89) $(GENCODE_SM90) $(GENCODE_SM120)
CC := nvcc

INCLUDE := 
LIB := 

SRC = mem_lat.cu

EXE = mem_lat

NVCC_FLGAS = -Xptxas -dlcm=cv -Xptxas -dscm=wt

release:
	$(CC) $(NVCC_FLGAS) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
