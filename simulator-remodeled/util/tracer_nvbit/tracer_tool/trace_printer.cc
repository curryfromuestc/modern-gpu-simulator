#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <map>
#include <limits>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iomanip>

// Include the generated Protocol Buffer headers
#include "../../traces_enhanced/pb_trace/include/trace.pb.h"
#include "../../traces_enhanced/pb_trace/include/cuda_stream.pb.h"
#include "../../traces_enhanced/pb_trace/include/kernel.pb.h"
#include "../../traces_enhanced/pb_trace/include/threadblock.pb.h"
#include "../../traces_enhanced/pb_trace/include/warp.pb.h"
#include "../../traces_enhanced/pb_trace/include/instruction.pb.h"
#include "../../traces_enhanced/pb_trace/include/address.pb.h"
#include "../../traces_enhanced/pb_trace/include/dim3d.pb.h"

// Function declarations
void clearScreen();
void displayMainMenu(const dynamic_trace::Trace& trace);
void deviceMenu(const dynamic_trace::Trace& trace);
void streamMenu(const dynamic_trace::Trace& trace, int64_t deviceId);
void kernelMenu(const dynamic_trace::Trace& trace, int64_t deviceId, int64_t streamId);
void kernelDetails(const dynamic_trace::Trace& trace, int64_t deviceId, int64_t streamId, int kernelIndex);
void threadblockDetails(const std::string& threadblockPath);
void warpDetails(const std::string& threadblockPath, int warpId);

std::string basePath;

// Helper function to find all threadblock files for a given stream and kernel
std::vector<std::string> findThreadblockFiles(int64_t deviceId, int64_t streamId, int64_t kernelId) {
    std::vector<std::string> files;
    
    // Update to use the new hierarchical folder structure
    std::string hierarchicalPath = basePath + "/threadblocks/device_" + std::to_string(deviceId) + 
                                  "/stream_" + std::to_string(streamId) + 
                                  "/kernel_" + std::to_string(kernelId);
    
    std::string prefix = "d_" + std::to_string(deviceId) + "_s_" + std::to_string(streamId) + "_k_" + std::to_string(kernelId) + "_";
    
    try {
        // Check if the hierarchical directory exists
        assert(std::filesystem::exists(hierarchicalPath));
        for (const auto& entry : std::filesystem::directory_iterator(hierarchicalPath)) {
            std::string filename = entry.path().filename().string();
            if (filename.find(prefix) == 0) {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error accessing threadblock directory: " << e.what() << std::endl;
    }
    
    return files;
}

// Helper function to parse block coordinates from filename
std::tuple<int, int, int> parseBlockCoordinates(const std::string& filename) {
    size_t lastUnderscore = filename.find_last_of('_');
    size_t dotPos = filename.find_last_of('.');
    if (lastUnderscore == std::string::npos || dotPos == std::string::npos) {
        return {-1, -1, -1};
    }
    
    std::string coordStr = filename.substr(lastUnderscore + 1, dotPos - lastUnderscore - 1);
    std::replace(coordStr.begin(), coordStr.end(), ',', ' ');
    std::stringstream ss(coordStr);
    
    int x, y, z;
    ss >> x >> y >> z;
    return {x, y, z};
}

// Helper function to clear input buffer
void clearInputBuffer() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: trace_printer <path>\n";
        return 1;
    }

    basePath = argv[1];
    std::string filePath = basePath + "/dynamic_trace.pb";

    // Open the file named dynamic_trace.pb in binary mode
    std::ifstream input(filePath, std::ios::in | std::ios::binary);
    if (!input) {
        std::cerr << "Error: dynamic_trace.pb not found in path: " << basePath << "\n";
        return 1;
    }

    // Create a Trace message and parse the file.
    dynamic_trace::Trace dyn_trace;
    if (!dyn_trace.ParseFromIstream(&input)) {
        std::cerr << "Error: Failed to parse dynamic_trace.pb\n";
        return 1;
    }
    input.close();

    // Start with the main menu
    displayMainMenu(dyn_trace);
    
    return 0;
}

void clearScreen() {
    // Use ANSI escape codes to clear screen (works on most terminals)
    std::cout << "\033[2J\033[1;1H";
}

void displayMainMenu(const dynamic_trace::Trace& trace) {
    while (true) {
        clearScreen();
        std::cout << "=== Trace Viewer Main Menu ===\n\n";
        std::cout << "General Trace Information:\n";
        std::cout << "  Trace Name\t\t\t: " << trace.name() << "\n";
        std::cout << "  Binary Version\t\t: " << trace.binary_version() << "\n";
        std::cout << "  Gathered Register Values\t: " << (trace.is_gathered_registers_values() ? "Yes" : "No") << "\n";
        std::cout << "  NVBIT Version\t\t\t: " << trace.nvbit_version() << "\n";
        std::cout << "  ACCELSIM Version\t\t: " << trace.accelsim_version() << "\n";
        std::cout << "  Number of Devices\t\t: " << trace.gpu_device().size() << "\n\n";

        std::cout << "Options:\n";
        std::cout << "1. View devices\n";
        std::cout << "-2. Exit program\n\n";
        std::cout << "Enter your choice: ";

        int choice;
        std::cin >> choice;
        
        if (std::cin.fail()) {
            clearInputBuffer();
            continue;
        }

        if (choice == -2) {
            return; // Exit
        }
        else if (choice == 1) {
            if (trace.gpu_device().empty()) {
                std::cout << "Error: No GPU devices found in trace.\n";
                std::cout << "Press Enter to continue...";
                clearInputBuffer();
                std::cin.get();
            } else {
                deviceMenu(trace);
            }
        }
    }
}

// New function to select a GPU device
void deviceMenu(const dynamic_trace::Trace& trace) {
    while (true) {
        clearScreen();
        std::cout << "=== Device Selection Menu ===\n\n";
        std::cout << "Available GPU devices:\n";
        
        for (const auto& device_pair : trace.gpu_device()) {
            int total_streams = device_pair.second.streams().size();
            std::cout << "Device ID: " << device_pair.first 
                      << " (Streams: " << total_streams << ")\n";
        }
        
        std::cout << "\nOptions:\n";
        std::cout << "Enter device ID to select, or:\n";
        std::cout << "-1. Go back to main menu\n";
        std::cout << "-2. Exit program\n\n";
        std::cout << "Enter your choice: ";
        
        int64_t device_id;
        std::cin >> device_id;
        
        if (std::cin.fail()) {
            clearInputBuffer();
            continue;
        }
        
        if (device_id == -2) {
            exit(0); // Exit
        }
        else if (device_id == -1) {
            return; // Go back
        }
        else {
            const auto& deviceMap = trace.gpu_device();
            auto itDevice = deviceMap.find(device_id);
            if (itDevice == deviceMap.end()) {
                std::cout << "Error: Device with id " << device_id << " not found.\n";
                std::cout << "Press Enter to continue...";
                clearInputBuffer();
                std::cin.get();
                continue;
            }
            
            streamMenu(trace, device_id);
        }
    }
}

void streamMenu(const dynamic_trace::Trace& trace, int64_t deviceId) {
    while (true) {
        clearScreen();
        std::cout << "=== Stream Selection Menu ===\n\n";
        std::cout << "Device ID: " << deviceId << "\n\n";
        std::cout << "Available streams:\n";
        
        const auto& deviceMap = trace.gpu_device();
        const dynamic_trace::gpu_device& selectedDevice = deviceMap.at(deviceId);
        
        for (const auto& stream_pair : selectedDevice.streams()) {
            std::cout << "Stream ID: " << stream_pair.first 
                      << " (Kernels: " << stream_pair.second.kernels_size() << ")\n";
        }
        
        std::cout << "\nOptions:\n";
        std::cout << "Enter stream ID to select, or:\n";
        std::cout << "-1. Go back to device menu\n";
        std::cout << "-2. Exit program\n\n";
        std::cout << "Enter your choice: ";
        
        int64_t stream_id;
        std::cin >> stream_id;
        
        if (std::cin.fail()) {
            clearInputBuffer();
            continue;
        }
        
        if (stream_id == -2) {
            exit(0); // Exit
        }
        else if (stream_id == -1) {
            return; // Go back
        }
        else {
            const auto& streamMap = selectedDevice.streams();
            auto itStream = streamMap.find(stream_id);
            if (itStream == streamMap.end()) {
                std::cout << "Error: Stream with id " << stream_id << " not found.\n";
                std::cout << "Press Enter to continue...";
                clearInputBuffer();
                std::cin.get();
                continue;
            }
            
            kernelMenu(trace, deviceId, stream_id);
        }
    }
}

void kernelMenu(const dynamic_trace::Trace& trace, int64_t deviceId, int64_t streamId) {
    const auto& deviceMap = trace.gpu_device();
    const dynamic_trace::gpu_device& selectedDevice = deviceMap.at(deviceId);
    const auto& streamMap = selectedDevice.streams();
    const dynamic_trace::cuda_stream& selectedStream = streamMap.at(streamId);
    
    while (true) {
        clearScreen();
        std::cout << "=== Kernel Selection Menu ===\n\n";
        std::cout << "Device ID: " << deviceId << "\n";
        std::cout << "Stream Information:\n";
        std::cout << "  Stream id    : " << selectedStream.id() << "\n";
        std::cout << "  # of Kernels : " << selectedStream.kernels_size() << "\n";
        std::cout << "  # of cuda events: " << selectedStream.ordered_cuda_events_size() << "\n\n";
        
        std::cout << "Available kernels in this stream:\n";
        for (int i = 0; i < selectedStream.kernels_size(); ++i) {
            const dynamic_trace::kernel& kern = selectedStream.kernels(i);
            std::cout << "id=" << kern.id() 
                      << ", name=\"" << kern.name() << "\"\n";
        }
        
        std::cout << "\nOptions:\n";
        std::cout << "Enter kernel index to select, or:\n";
        std::cout << "-1. Go back to stream selection\n";
        std::cout << "-2. Exit program\n\n";
        std::cout << "Enter your choice: ";
        
        int kernelIndex;
        std::cin >> kernelIndex;
        
        if (std::cin.fail()) {
            clearInputBuffer();
            continue;
        }
        
        if (kernelIndex == -2) {
            exit(0); // Exit
        }
        else if (kernelIndex == -1) {
            return; // Go back
        }
        else if (kernelIndex <= 0 || kernelIndex > selectedStream.kernels_size()) {
            std::cout << "Error: Kernel index out of range.\n";
            std::cout << "Press Enter to continue...";
            clearInputBuffer();
            std::cin.get();
            continue;
        }
        else {
            kernelDetails(trace, deviceId, streamId, kernelIndex-1);
        }
    }
}

void kernelDetails(const dynamic_trace::Trace& trace, int64_t deviceId, int64_t streamId, int kernelIndex) {
    const auto& deviceMap = trace.gpu_device();
    const dynamic_trace::gpu_device& selectedDevice = deviceMap.at(deviceId);
    const auto& streamMap = selectedDevice.streams();
    const dynamic_trace::cuda_stream& selectedStream = streamMap.at(streamId);
    const dynamic_trace::kernel& selKernel = selectedStream.kernels(kernelIndex);
    
    while (true) {
        clearScreen();
        std::cout << "=== Kernel Details ===\n\n";
        std::cout << "Device ID: " << deviceId << "\n";
        std::cout << "Stream ID: " << streamId << "\n";
        std::cout << "General Kernel Information:\n";
        std::cout << "  Kernel id                 : " << selKernel.id() << "\n";
        std::cout << "  Kernel name               : " << selKernel.name() << "\n";
        std::cout << "  Function Unique id        : " << selKernel.function_unique_id() << "\n";
        std::cout << "  Shared Memory Size        : " << selKernel.size_shared_memory() << "\n";
        std::cout << "  Number of Registers       : " << selKernel.number_of_registers() << "\n";
        std::cout << "  Shared Memory Base Address: 0x" << std::hex << selKernel.shared_memory_base_address() << std::dec << "\n";
        std::cout << "  Local Memory Base Address : 0x" << std::hex << selKernel.local_memory_base_address() << std::dec << "\n";

        // Display grid_dim and block_dim information.
        const dynamic_trace::dim3d& grid = selKernel.grid_dim();
        const dynamic_trace::dim3d& block = selKernel.block_dim();
        std::cout << "  Grid Dim: (" << grid.x() << ", " << grid.y() << ", " << grid.z() << ")\n";
        std::cout << "  Block Dim: (" << block.x() << ", " << block.y() << ", " << block.z() << ")\n";
        
        std::cout << "\nEnter threadblock coordinates in format x,y,z (or -1 to go back, -2 to exit): ";
        
        std::string input;
        std::cin >> input;
        
        if (std::cin.fail()) {
            clearInputBuffer();
            continue;
        }
        
        if (input == "-2") {
            exit(0); // Exit program
        }
        else if (input == "-1") {
            return; // Go back
        }
        else {
            // Parse x,y,z coordinates
            std::replace(input.begin(), input.end(), ',', ' ');
            std::stringstream ss(input);
            int x, y, z;
            
            if (ss >> x >> y >> z) {
                // Validate coordinates
                if (x < 0 || x >= grid.x() || y < 0 || y >= grid.y() || z < 0 || z >= grid.z()) {
                    std::cout << "\nInvalid coordinates. Must be within grid dimensions.\n";
                    std::cout << "Press Enter to continue...";
                    clearInputBuffer();
                    std::cin.get();
                    continue;
                }
                
                // Construct the TB ID string that appears in the filename
                std::string tb_string_id = "d_" + std::to_string(deviceId) + 
                                          "_s_" + std::to_string(streamId) + 
                                          "_k_" + std::to_string(selKernel.id()) + "_" + 
                                          std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
                
                // First try the hierarchical path
                std::string hierarchicalPath = basePath + "/threadblocks/device_" + std::to_string(deviceId) + 
                                              "/stream_" + std::to_string(streamId) + 
                                              "/kernel_" + std::to_string(selKernel.id()) + "/" + 
                                              tb_string_id + ".pb";
                
                // Check if the hierarchical path file exists
                std::ifstream testHierarchical(hierarchicalPath);
                if (testHierarchical.good()) {
                    testHierarchical.close();
                    threadblockDetails(hierarchicalPath);
                } else {
                    // If we get here, neither path worked
                    std::cout << "\nThreadblock file not found!\n";
                    std::cout << "Tried path:  " << hierarchicalPath << std::endl;
                    std::cout << "Press Enter to continue...";
                    clearInputBuffer();
                    std::cin.get();
                }
            } else {
                std::cout << "\nInvalid format. Please use format x,y,z (e.g. 0,0,0)\n";
                std::cout << "Press Enter to continue...";
                clearInputBuffer();
                std::cin.get();
            }
        }
    }
}

void threadblockDetails(const std::string& threadblockPath) {
    clearScreen();
    std::cout << "=== Threadblock Details ===\n\n";
    std::cout << "Threadblock file: " << std::filesystem::path(threadblockPath).filename().string() << "\n\n";
    
    // Open and parse the threadblock protocol buffer file
    std::ifstream input(threadblockPath, std::ios::in | std::ios::binary);
    if (!input) {
        std::cout << "Error: Could not open threadblock file.\n";
        std::cout << "\nPress Enter to return...";
        clearInputBuffer();
        std::cin.get();
        return;
    }
    
    dynamic_trace::threadblock threadblockData;
    if (!threadblockData.ParseFromIstream(&input)) {
        std::cout << "Error: Failed to parse threadblock file.\n";
        input.close();
        std::cout << "\nPress Enter to return...";
        clearInputBuffer();
        std::cin.get();
        return;
    }
    input.close();
    
    // Extract coordinates from filename
    auto filename = std::filesystem::path(threadblockPath).filename().string();
    auto [blockX, blockY, blockZ] = parseBlockCoordinates(filename);
    
    std::cout << "Block coordinates: (" << blockX << ", " << blockY << ", " << blockZ << ")\n";
    
    // Display threadblock information
    std::cout << "Warp count: " << threadblockData.warps_size() << "\n\n";
    
    // Count total instructions across all warps
    int totalInstructions = 0;
    std::cout << "Available warps:\n";
    for (const auto& warp : threadblockData.warps()) {
        std::cout << "Warp " << warp.first << ": " << warp.second.instructions_size() << " instructions\n";
        totalInstructions += warp.second.instructions_size();
    }
    std::cout << "Total threadblock warp instructions: " << totalInstructions << "\n\n";
    
    // Add option to view specific warp
    std::cout << "Options:\n";
    std::cout << "Enter warp ID to see details, or:\n";
    std::cout << "-1. Go back\n\n";
    std::cout << "Enter your choice: ";
    
    int warpChoice;
    std::cin >> warpChoice;
    
    if (std::cin.fail()) {
        clearInputBuffer();
        return;
    }
    
    if (warpChoice == -1) {
        return; // Go back
    } else {
        // Check if the selected warp exists
        if (threadblockData.warps().find(warpChoice) != threadblockData.warps().end()) {
            warpDetails(threadblockPath, warpChoice);
        } else {
            std::cout << "\nError: Warp ID " << warpChoice << " not found.\n";
            std::cout << "Press Enter to continue...";
            clearInputBuffer();
            std::cin.get();
        }
    }
}

// New function to display details of a specific warp
void warpDetails(const std::string& threadblockPath, int warpId) {
    clearScreen();
    std::cout << "=== Warp Details ===\n\n";
    std::cout << "Threadblock: " << std::filesystem::path(threadblockPath).filename().string() << "\n";
    std::cout << "Warp ID: " << warpId << "\n\n";
    
    // Open and parse the threadblock file
    std::ifstream input(threadblockPath, std::ios::in | std::ios::binary);
    if (!input) {
        std::cout << "Error: Could not open threadblock file.\n";
        std::cout << "\nPress Enter to return...";
        clearInputBuffer();
        std::cin.get();
        return;
    }
    
    dynamic_trace::threadblock threadblockData;
    if (!threadblockData.ParseFromIstream(&input)) {
        std::cout << "Error: Failed to parse threadblock file.\n";
        input.close();
        std::cout << "\nPress Enter to return...";
        clearInputBuffer();
        std::cin.get();
        return;
    }
    input.close();
    
    // Find the requested warp in the map
    const auto& warpsMap = threadblockData.warps();
    auto warpIter = warpsMap.find(warpId);
    
    if (warpIter == warpsMap.end()) {
        std::cout << "Error: Warp ID " << warpId << " not found in threadblock.\n";
    } else {
        const dynamic_trace::warp& warp = warpIter->second;
        
        // Display warp information
        std::cout << "Total instructions: " << warp.instructions_size() << "\n\n";
        
        // Display all instructions with PC and threadmask in hexadecimal
        std::cout << std::left 
                  << std::setw(5) << "Idx" 
                  << std::setw(10) << "PC" 
                  << std::setw(15) << "Thread Mask" 
                  << "Opcode\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (int i = 0; i < warp.instructions_size(); i++) {
            const auto& instr = warp.instructions(i);
            
            // Format PC as hexadecimal with 0x prefix and leading zeros
            std::stringstream pcHex;
            pcHex << "0x" << std::hex << instr.pc() <<std::dec;
            
            // // Format thread mask as hexadecimal with 0x prefix
            std::stringstream maskHex;
            maskHex << "0x";
            uint32_t thread_mask = instr.active_mask() & instr.predicate_mask();
            maskHex << "0x" << std::hex << std::setfill('0') << std::setw(8) << thread_mask << std::dec;

            std::cout << std::left 
                      << std::setw(5) << i 
                      << std::setw(10) << pcHex.str() 
                      << std::setw(15) << maskHex.str() << "\n";
        }
    }
    
    std::cout << "\nPress Enter to return to threadblock details...";
    clearInputBuffer();
    std::cin.get();
}