NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

DEBUG_TOOL?=0

OPTFLAGS=

ifneq ($(DEBUG_TOOL),1)
	OPTFLAGS +=-O3
else
	OPTFLAGS +=-g -O0
endif

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif


PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
	MAXRREGCOUNT_FLAG=-maxrregcount=24
else
	MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../nvbit_release/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)
ARCH?=all

ARCH_VER_REQ=11.7 # NVCC < 11.7 doesn't support -arch=all flag
ARCH_VER_CHECK=$(shell echo "${NVCC_VER} >= $(ARCH_VER_REQ)" | bc)

ifeq ($(ARCH_VER_CHECK),0)
ifeq ($(ARCH), all)
$(shell echo "${ARCH}, ${ARCH_VER_CHECK}, ${NVCC_VER}" )
$(error ERROR: ARCH environment variable must be set to "sm_XX" for nvcc < 11.7, ex: run "export ARCH=sm_70 for compute capability 70")
endif
endif

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

# MOD. Begin. ENHANCED TRACES
TRACES_ENHANCED_DIR=../../traces_enhanced/
TRACES_ENHANCED_PROTOCOL_BUFFER_DIR=../../traces_enhanced/pb_trace
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)
TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ=$(wildcard $(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/obj/*.o)
TRACE_PRINTER = trace_printer
# MOD. End. ENHANCED TRACES


all: traces_enhanced $(NVBIT_TOOL) trace_printer

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a traces_enhanced
	$(NVCC) -arch=$(ARCH) $(OPTFLAGS) $(OBJECTS) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -lprotobuf -shared -o $@

%.o: %.cu common.h
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(OPTFLAGS) -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# MOD. Begin. ENHANCED TRACES
traces_enhanced:
	$(MAKE) -C $(TRACES_ENHANCED_DIR)
# MOD. End. ENHANCED TRACES

# Build the trace_printer executable
# First compile trace_printer's code into a shared library
trace_printer: traces_enhanced
	$(NVCC) -std=c++17 $(OPTFLAGS) -o $(TRACE_PRINTER) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) trace_printer.cc -lprotobuf

clean:
	rm -f *.so *.o trace_printer
	$(MAKE) clean -C $(TRACES_ENHANCED_DIR)
