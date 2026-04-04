CC=g++
MKDIR_P = mkdir -p

ODIR=obj
SRCDIR=src

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -O0 -g3 -fPIC -std=c++17
else ifeq ($(DEBUG_TOOL),1)
	CXXFLAGS = -Wall -O0 -g3 -fPIC -std=c++17
else
	CXXFLAGS = -Wall -O3 -g3 -fPIC -std=c++17
endif

TRACES_ENHANCED_PROTO_DIR=dynamic_trace
TRACES_ENHANCED_DIR=.
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)

# Protocol Buffer setup
PROTOC=protoc
PROTO_FILES=$(wildcard $(TRACES_ENHANCED_PROTO_DIR)/*.proto)

# Define directories for generated files
PROTO_GEN_DIR=$(TRACES_ENHANCED_DIR)/pb_trace
PROTO_SRC_DIR=$(PROTO_GEN_DIR)/src
PROTO_HDR_DIR=$(PROTO_GEN_DIR)/include
PROTO_OBJ_DIR=$(PROTO_GEN_DIR)/obj

# Add protobuf includes and libraries
PROTOBUF_INCLUDES=-I/home/yanggl/miniconda3/include
PROTOBUF_LIBS=-L/home/yanggl/miniconda3/lib -lprotobuf

# Create directory structure if it doesn't exist
$(shell mkdir -p $(PROTO_SRC_DIR) $(PROTO_HDR_DIR) $(PROTO_OBJ_DIR))

# Define path transformations for proto files
PROTO_SRCS=$(patsubst $(TRACES_ENHANCED_PROTO_DIR)/%.proto,$(PROTO_SRC_DIR)/%.pb.cc,$(PROTO_FILES))
PROTO_HDRS=$(patsubst $(TRACES_ENHANCED_PROTO_DIR)/%.proto,$(PROTO_HDR_DIR)/%.pb.h,$(PROTO_FILES))
PROTO_OBJS=$(patsubst $(PROTO_SRC_DIR)/%.pb.cc,$(PROTO_OBJ_DIR)/%.pb.o,$(PROTO_SRCS))

# Add includes for generated headers
INCLUDES+=$(PROTOBUF_INCLUDES) -I$(TRACES_ENHANCED_DIR) -I$(PROTO_HDR_DIR)
LIBS+=$(PROTOBUF_LIBS)


CFLAGS=$(CXXFLAGS) -I$(CUDA_INSTALL_PATH)/include -fPIC

LIBS=-lz -pthread

SOURCES := $(shell find $(SRCDIR) -name '*.cc')
OBJ := $(patsubst $(SRCDIR)/%,$(ODIR)/%,$(SOURCES:.cc=.o))

.PHONY: clean depend

traces_enhanced: directories protobuf depend

directories: ${ODIR}

${ODIR}:
	${MKDIR_P} ${ODIR}

depend: $(OBJ)

$(ODIR)/%.o: $(SRCDIR)/%.cc
	$(CC) -c -o $@ $< $(CFLAGS)

# Protocol buffer compilation rules
$(PROTO_SRC_DIR)/%.pb.cc $(PROTO_HDR_DIR)/%.pb.h: $(TRACES_ENHANCED_PROTO_DIR)/%.proto
	$(PROTOC) --cpp_out=$(PROTO_SRC_DIR) --proto_path=$(TRACES_ENHANCED_PROTO_DIR) $<
	@mv $(PROTO_SRC_DIR)/$*.pb.h $(PROTO_HDR_DIR)/$*.pb.h

$(PROTO_OBJ_DIR)/%.pb.o: $(PROTO_SRC_DIR)/%.pb.cc $(PROTO_HDR_DIR)/%.pb.h
	$(CXX) -c $(INCLUDES) $(PROTOBUF_INCLUDES) -fPIC -o $@ $<


# Generate Protocol Buffer files
protobuf: $(PROTO_SRCS) $(PROTO_HDRS) $(PROTO_OBJS)
	@echo "Protocol buffer files generated"
# MOD. End. ENHANCED TRACES

clean:
	rm -rf $(ODIR)
	rm -rf $(PROTO_GEN_DIR)