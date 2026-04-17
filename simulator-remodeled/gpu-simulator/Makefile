# Copyright (c) 2020 Timothy Rogers, Purdue University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


BIN_DIR=./bin/$(ACCELSIM_CONFIG)
BUILD_DIR=./build/$(ACCELSIM_CONFIG)

# MOD. Begin. ENHANCED TRACES
TRACES_ENHANCED_DIR=../util/traces_enhanced
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)
TRACES_ENHANCED_SRC=$(wildcard $(TRACES_ENHANCED_DIR)/src)
TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/pb_trace/obj/*.o)
# MOD. End. ENHANCED TRACES

include version_detection.mk

ifeq ($(ACCELSIM_CONFIG), debug)
	export DEBUG=1
else
	export DEBUG=0
endif

CXXFLAGS = -fopenmp

ifeq ($(DEBUG),1)
	CXXFLAGS += -Wall -O0 -g3 -fPIC -std=c++17
else
	CXXFLAGS += -Wall -O3 -g3 -fPIC -std=c++17 
endif

SANITIZER_FLAG_FULL=
SANITZER_FLAG=

ifeq ($(SANITIZER_COM_FLAG), addr_sanitizer_enabled)
	SANITIZER_FLAG_FULL = -fsanitize=address,undefined -static-libasan
	SANITZER_FLAG = -fsanitize=address,undefined
endif

ifeq ($(SANITIZER_COM_FLAG), tsan_sanitizer_enabled)
	SANITIZER_FLAG_FULL = -fsanitize=thread -static-libtsan
	SANITZER_FLAG = -fsanitize=thread
endif

all: $(BIN_DIR)/accel-sim.out

$(BUILD_DIR)/main.makedepend: depend makedirs

makedirs:
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi;
	if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi;

gpgpu-sim: checkenv makedirs
	$(MAKE) -C gpgpu-sim

checkenv: makedirs
	 @if [ ! -n "$(ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN)" ]; then \
		echo "ERROR *** run 'source setup_environment.sh' before 'make'; please see README."; \
		exit 1; \
	fi

$(BIN_DIR)/accel-sim.out: traces_enhanced trace-driven trace-parser gpgpu-sim makedirs $(BUILD_DIR)/main.o version
	$(CXX) -std=c++17 -L$(GPGPUSIM_ROOT)/lib/$(GPGPUSIM_CONFIG)/ -lm -lGL -pthread -lz -lcudart -lz $(BUILD_DIR)/*.o $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) -lgomp -lprotobuf -o $(BIN_DIR)/accel-sim.out $(SANITIZER_FLAG_FULL)

$(BUILD_DIR)/main.o: main.cc version
	$(CXX) $(CXXFLAGS) $(SANITZER_FLAG) -I$(BUILD_DIR) -I./trace-driven -I./trace-parser -I$(GPGPUSIM_ROOT)/libcuda -I$(GPGPUSIM_ROOT)/src -I$(CUDA_INSTALL_PATH)/include -c main.cc -o $(BUILD_DIR)/main.o

version:
	echo "const char *g_accelsim_version=\"$(ACCELSIM_BUILD)\";" > $(BUILD_DIR)/accelsim_version.h

depend: checkenv makedirs
	touch $(BUILD_DIR)/main.makedepend
	makedepend -f$(BUILD_DIR)/main.makedepend -p$(BUILD_DIR)/ main.cc 2> /dev/null

trace-driven: checkenv makedirs traces_enhanced
	$(MAKE) -C trace-driven depend
	$(MAKE) -C trace-driven

trace-parser: checkenv makedirs traces_enhanced
	$(MAKE) -C trace-parser depend
	$(MAKE) -C trace-parser

# MOD. Begin. ENHANCED TRACES
traces_enhanced: checkenv makedirs
	$(MAKE) -C $(TRACES_ENHANCED_DIR)
# MOD. End. ENHANCED TRACES

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(BUILD_DIR)
	$(MAKE) clean -C gpgpu-sim
	$(MAKE) clean -C $(TRACES_ENHANCED_DIR)


include $(BUILD_DIR)/main.makedepend
