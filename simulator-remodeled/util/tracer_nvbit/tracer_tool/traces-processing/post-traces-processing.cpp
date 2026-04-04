#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include "../../../traces_enhanced/src/traced_execution.h"
using namespace std;

struct threadblock_info {
  bool initialized;
  unsigned tb_id_x, tb_id_y, tb_id_z;
  vector<vector<string>> warp_insts_array;
  threadblock_info() {
    initialized = false;
    tb_id_x = tb_id_y = tb_id_z = 0;
  }
};

void group_per_block(const char *filepath);
void group_per_core(const char *filepath);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Add a flag for LDGSTS instruction to indicate which one to remove
vector<vector<bool>> ldgsts_flags;  // true to remove, false to not
vector<vector<std::string>> ldgsts_shmem_addrs;  // true to remove, false to not
traced_execution extra_trace_information;

int main(int argc, char **argv) {

  string kernellist_filepath;
  bool is_per_core;
  if (argc == 1) {
    cout << "File path is missing\n";
    return 0;
  } else if (argc == 2) {
    kernellist_filepath = argv[1];
    is_per_core = true;

  } else if (argc == 3) {
    kernellist_filepath = argv[1];
    is_per_core = bool(argv[2]);
  } else {
    cout << "Too Many Arguemnts!\n";
    return 0;
  }

  ifstream ifs;
  ofstream ofs;

  ifs.open(kernellist_filepath.c_str());
  ofs.open((string(kernellist_filepath) + ".g").c_str());

  if (!ifs.is_open()) {
    cout << "Unable to open file: " << kernellist_filepath << endl;
    return 0;
  }

  string directory(kernellist_filepath);
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string extra_trace_information_path = directory + "/extra_info/enhanced_execution_info.json";

  extra_trace_information.DeserializeFromFile(extra_trace_information_path.c_str());

  string line;
  string filepath;
  while (!ifs.eof()) {
    getline(ifs, line);
    if (line.empty())
      continue;
    else if (line.substr(0, 6) == "Memcpy") {
      ofs << line << endl;
    } else if (line.substr(0, 6) == "kernel") {
      filepath = directory + "/" + line;
      group_per_block(filepath.c_str());
      ofs << line + "g" << endl;
    } else {
      cout << "Undefined command: " << line << endl;
      return 0;
    }
  }

  ifs.close();
  ofs.close();
  return 0;
}

void group_per_block(const char *filepath) {

  ofstream ofs;
  ifstream ifs;

  ifs.open(filepath);

  if (!ifs.is_open()) {
    cout << "Unable to open file: " << filepath << endl;
    return;
  }

  cout << "Processing file " << filepath << endl;
  ofs.open((string(filepath) + "g").c_str());

  vector<threadblock_info> insts;
  unsigned grid_dim_x, grid_dim_y, grid_dim_z, tb_dim_x, tb_dim_y, tb_dim_z;
  unsigned tb_id_x, tb_id_y, tb_id_z, tb_id, warpid_tb;
  string line;
  stringstream ss;
  string string1, string2;
  bool found_grid_dim = false, found_block_dim = false;

  std::string kernel_name;

  while (!ifs.eof()) {
    getline(ifs, line);

    if (line.length() == 0 || line[0] == '#') {
      ofs << line << endl;
      continue;
    }

    else if (line[0] == '-') {
      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;
      if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &grid_dim_x, &grid_dim_y,
               &grid_dim_z);
        found_grid_dim = true;
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &tb_dim_x, &tb_dim_y,
               &tb_dim_z);
        found_block_dim = true;
      }else if (string1 == "kernel" && string2 == "name") {
        std::stringstream ss_temp(line);
        ss_temp.ignore();
        std::string dummy1, dummy2;
        ss_temp >> dummy1 >> dummy2;
        ss_temp.ignore();
        ss_temp.ignore();
        ss_temp >> kernel_name;
      }

      if (found_grid_dim && found_block_dim) {
        insts.resize(grid_dim_x * grid_dim_y * grid_dim_z);

        // Size the ldgsts_flags vector
        ldgsts_flags.resize(grid_dim_x * grid_dim_y * grid_dim_z);
        ldgsts_shmem_addrs.resize(grid_dim_x * grid_dim_y * grid_dim_z);

        for (unsigned i = 0; i < insts.size(); ++i) {
          insts[i].warp_insts_array.resize(
              ceil(float(tb_dim_x * tb_dim_y * tb_dim_z) / 32));

          // Size the ldgsts_flags vector
          ldgsts_flags[i].resize(ceil(float(tb_dim_x * tb_dim_y * tb_dim_z) / 32));
          ldgsts_shmem_addrs[i].resize(ceil(float(tb_dim_x * tb_dim_y * tb_dim_z) / 32));
          for (unsigned j = 0; j < ldgsts_flags[i].size(); j++) {
            ldgsts_flags[i][j] = true;
            ldgsts_shmem_addrs[i][j] = "";
          }
        }
      }
      ofs << line << endl;
      continue;
    } else {

      ss.str(line);
      ss >> tb_id_x >> tb_id_y >> tb_id_z >> warpid_tb;
      tb_id =
          tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
      if (!insts[tb_id].initialized) {
        insts[tb_id].tb_id_x = tb_id_x;
        insts[tb_id].tb_id_y = tb_id_y;
        insts[tb_id].tb_id_z = tb_id_z;
        insts[tb_id].initialized = true;
      }
	    // ss.ignore(); //remove the space
	    // rest_of_line.clear();
      // getline(ss, rest_of_line); //get rest of the string!
      std::string rest_of_line(ss.str().substr(ss.tellg() + std::streampos(1)));

      // Collide LDGSTS instructions
      std::string opcode;
      std::stringstream trace_ss;
      std::string pc_str, mask_str, num_refs, unique_function_id_str;
      trace_ss << rest_of_line;

      trace_ss >> pc_str >> mask_str >> unique_function_id_str;
      unsigned int pc_num = std::stoul(pc_str, nullptr, 16);
      unsigned int unique_function_id_num = std::stoul(unique_function_id_str, nullptr, 10);

      bool is_ldgst = false;
      if(extra_trace_information.has_kernel_with_unique_function_id(unique_function_id_num)) {
        opcode = extra_trace_information.get_kernel_by_unique_function_id(unique_function_id_num).get_instruction(pc_num).get_op_code();
        is_ldgst = opcode.find("LDGSTS") != string::npos;
      }
      

      // One actual LDGSTS instruction includes 2 LDGSTS instructions in the trace, 
      // because it has two memory references. 
      // This is trying to remove the one with the shared memory address and collide the information with the one of global memory
      if (is_ldgst) {
        trace_ss >> num_refs;
        std::string addr_str(trace_ss.str().substr(trace_ss.tellg() + std::streampos(1)));
        if (!ldgsts_flags[tb_id][warpid_tb]) {
          insts[tb_id].warp_insts_array[warpid_tb].push_back(rest_of_line + ldgsts_shmem_addrs[tb_id][warpid_tb]);
        }
        ldgsts_flags[tb_id][warpid_tb] = !ldgsts_flags[tb_id][warpid_tb];
        ldgsts_shmem_addrs[tb_id][warpid_tb] = addr_str;
      }else {
        insts[tb_id].warp_insts_array[warpid_tb].push_back(rest_of_line);
      }
    }
  }

  for (unsigned i = 0; i < insts.size(); ++i) {
    // ofs<<string<<endl;
    if (insts[i].initialized && insts[i].warp_insts_array.size() > 0) {
      ofs << endl << "#BEGIN_TB" << endl;
      ofs << endl
          << "thread block = " << insts[i].tb_id_x << "," << insts[i].tb_id_y
          << "," << insts[i].tb_id_z << endl;
    } else {
      cout << "Warning: Thread block " << insts[i].tb_id_x << ","
           << insts[i].tb_id_y << "," << insts[i].tb_id_z << " is empty"
           << endl;
      continue;
      // ofs.close();
      // return;
    }
    for (unsigned j = 0; j < insts[i].warp_insts_array.size(); ++j) {
      ofs << endl << "warp = " << j << endl;
      ofs << "insts = " << insts[i].warp_insts_array[j].size() << endl;
      if (insts[i].warp_insts_array[j].size() == 0) {
        cout << "Warning: Warp " << j << " in thread block" << insts[i].tb_id_x
             << "," << insts[i].tb_id_y << "," << insts[i].tb_id_z
             << " is empty" << endl;
        //	ofs.close();
        //	return;
      }
      for (unsigned k = 0; k < insts[i].warp_insts_array[j].size(); ++k) {
        ofs << insts[i].warp_insts_array[j][k] << endl;
      }
    }
    ofs << endl << "#END_TB" << endl;
  }

  ofs.close();
  ifs.close();
}

void group_per_core(const char *filepath) {

  // TO DO
}
