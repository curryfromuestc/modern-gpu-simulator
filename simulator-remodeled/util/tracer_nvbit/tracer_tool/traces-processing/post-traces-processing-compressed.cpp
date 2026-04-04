#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <assert.h>
#include <tuple>
#include <zlib.h>
#include <cstring>

#include "../../../traces_enhanced/src/traced_execution.h"

using namespace std;
namespace fs = std::filesystem;

void process_raw_kernel_trace(const fs::path &trace_filepath);

class Kernel {
public:
  Kernel(const fs::path &kernel_dirpath, const fs::path &raw_trace_filepath, unsigned long long max_RAM, traced_execution *extra_trace_info) 
          : m_tot_insts_bytes(0), m_kernel_dirpath(kernel_dirpath), m_max_RAM(max_RAM), m_raw_trace_filepath(raw_trace_filepath) {
            m_extra_trace_info = extra_trace_info;
            m_raw_trace_file.open(m_raw_trace_filepath);
            if (!m_raw_trace_file) {
              cerr << "Error: cannot open file " << raw_trace_filepath << std::endl;
              exit(1);
            }

            ofstream kernel_tfile_header(kernel_dirpath / "header");
            if (!kernel_tfile_header) {
              cerr << "Error: cannot create file " << (kernel_dirpath / "header") << endl;
              exit(1); 
            }
            
            while(!m_raw_trace_file.eof()) {

              string line;              
              getline(m_raw_trace_file, line);
              if(line[0] == '#') {
                getline(m_raw_trace_file, line);
                kernel_tfile_header << "#traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses] dest_pred_num [reg_pred_dests] pred_src_num [reg_pred_srcs] pred_reg" << endl;
                break;
              }
              
              kernel_tfile_header << line << std::endl;
              
              if (line == "") continue;

              if (line.substr(0, 13) == "-grid dim = ("s) {
                istringstream iss(line.substr(13));

                char ch;
                iss >> m_grid_dim_x >> ch;
                iss >> m_grid_dim_y >> ch;
                iss >> m_grid_dim_z >> ch;

                continue;
              }

              if (line.substr(0, 14) == "-block dim = ("s) {
                istringstream iss(line.substr(14));   
                char ch;
                iss >> m_block_dim_x >> ch;
                iss >> m_block_dim_y >> ch;
                iss >> m_block_dim_z >> ch;
                continue;
              } 

              if (line.substr(0, 15) == "-kernel name = "s) {
                m_kernel_name = line.substr(15);
                continue;
              }
            }

            kernel_tfile_header.close();
            
            unsigned  int n_threads_per_block = m_block_dim_x * m_block_dim_y * m_block_dim_z;
            m_n_warps_per_block = (n_threads_per_block + 31) / 32; // the ceil of the division by warp size
            unsigned n_blocks_in_grid = m_grid_dim_x * m_grid_dim_y * m_grid_dim_z;
            unsigned long long tot_warps = n_blocks_in_grid * m_n_warps_per_block;
            m_insts = new std::vector<std::vector<std::vector<std::string>>>(n_blocks_in_grid);
            // Size the ldgsts_flags vector
            m_ldgsts_flags.resize(n_blocks_in_grid);
            m_ldgsts_shmem_addrs.resize(n_blocks_in_grid);

            m_tb_index.resize(n_blocks_in_grid);
            m_tb_initialized.resize(n_blocks_in_grid, false);
            m_n_tot_insts_per_warp.resize(tot_warps, 0);
            for (unsigned i = 0; i < (*m_insts).size(); ++i) {
              (*m_insts)[i].resize(m_n_warps_per_block);
              m_ldgsts_flags[i].resize(m_n_warps_per_block);
              m_ldgsts_shmem_addrs[i].resize(m_n_warps_per_block);
              for (unsigned j = 0; j < m_ldgsts_flags[i].size(); j++) {
                m_ldgsts_flags[i][j] = true;
                m_ldgsts_shmem_addrs[i][j] = "";
              }
            }
            // for (auto &block : (*m_insts)) {
            //   block.resize(m_n_warps_per_block);
            // }

          }

  void process_raw_trace_file() {
    assert(m_raw_trace_file.is_open());
    while(!m_raw_trace_file.eof()) {
      string line;
      getline(m_raw_trace_file, line);
      if(line == "") continue; 
      istringstream iss(line);
      unsigned tb_id_x, tb_id_y, tb_id_z, wid;
      iss >> tb_id_x >> tb_id_y >> tb_id_z >> wid >> ws;
      unsigned tb_id = tb_id_z * m_grid_dim_y * m_grid_dim_x + tb_id_y * m_grid_dim_x + tb_id_x;
      if(!m_tb_initialized[tb_id]) {
        m_tb_index[tb_id] = make_tuple(tb_id_x, tb_id_y, tb_id_z);
        m_tb_initialized[tb_id] = true;
      }

      string rest_of_line;
      getline(iss, rest_of_line);
      assert(m_insts);

      // Collide LDGSTS instructions
      std::string opcode;
      std::stringstream trace_ss;
      std::string pc_str, mask_str, num_refs, unique_function_id_str;
      trace_ss << rest_of_line;

      trace_ss >> pc_str >> mask_str >> unique_function_id_str;
      unsigned int pc_num = std::stoul(pc_str, nullptr, 16);
      unsigned int unique_function_id_num = std::stoul(unique_function_id_str, nullptr, 10);
      
      bool is_ldgst = false;
      if(m_extra_trace_info->has_kernel_with_unique_function_id(unique_function_id_num)) {
        opcode = m_extra_trace_info->get_kernel_by_unique_function_id(unique_function_id_num).get_instruction(pc_num).get_op_code();
        is_ldgst = opcode.find("LDGSTS") != string::npos;
      }

      // One actual LDGSTS instruction includes 2 LDGSTS instructions in the trace, 
      // because it has two memory references. 
      // This is trying to remove the one with the shared memory address and collide the information with the one of global memory
      if (is_ldgst) {
        trace_ss >> num_refs;
        std::string addr_str(trace_ss.str().substr(trace_ss.tellg() + std::streampos(1)));;
        if (!m_ldgsts_flags[tb_id][wid]) {
          (*m_insts)[tb_id][wid].push_back(rest_of_line + m_ldgsts_shmem_addrs[tb_id][wid]);
        }
        m_ldgsts_flags[tb_id][wid] = !m_ldgsts_flags[tb_id][wid];
        m_ldgsts_shmem_addrs[tb_id][wid] = addr_str;
      }else {
        (*m_insts)[tb_id][wid].push_back(rest_of_line);
      }
      // (*m_insts)[tb_id][wid].push_back(rest_of_line);
      m_tot_insts_bytes += rest_of_line.size();
      if(m_tot_insts_bytes > m_max_RAM) {
        flush();
      }
    }
    processing_finished();
  }

  void flush() {
    // write the processed instructions to the file
    for (unsigned tb_idx = 0; tb_idx < m_insts->size(); ++tb_idx) {
      for (unsigned tb_w_idx = 0; tb_w_idx < (*m_insts)[tb_idx].size(); ++tb_w_idx) {
        auto &insts = (*m_insts)[tb_idx][tb_w_idx];
        if (insts.size()) {
          unsigned global_w_idx = tb_idx * m_n_warps_per_block + tb_w_idx;
          fs::path w_otrace_filepath = m_kernel_dirpath / to_string(global_w_idx);
          ofstream w_otrace_file(w_otrace_filepath, ios::app);
          if(!w_otrace_file.is_open()) {
            cerr << "Error: cannot open file " << w_otrace_filepath << std::endl;
            exit(1);
          }
          unsigned long long insts_in_warp = 0;
          for(auto &inst : insts) {
            w_otrace_file << inst << std::endl;
            insts_in_warp++;
          }
          m_n_tot_insts_per_warp[global_w_idx] += insts_in_warp;
          //insts.clear();
          //std::vector<string>().swap(insts); // release the memory
          w_otrace_file.close();
        }
      }
    }
    m_tot_insts_bytes = 0;
    delete m_insts;
    
    
    unsigned n_blocks_in_grid = m_grid_dim_x * m_grid_dim_y * m_grid_dim_z;
    m_insts = new std::vector<std::vector<std::vector<std::string>>>(n_blocks_in_grid);

    // Size the ldgsts_flags vector
    m_ldgsts_flags.resize(n_blocks_in_grid);
    m_ldgsts_shmem_addrs.resize(n_blocks_in_grid);

    for (unsigned i = 0; i < (*m_insts).size(); ++i) {
      (*m_insts)[i].resize(m_n_warps_per_block);
      m_ldgsts_flags[i].resize(m_n_warps_per_block);
      m_ldgsts_shmem_addrs[i].resize(m_n_warps_per_block);
      for (unsigned j = 0; j < m_ldgsts_flags[i].size(); j++) {
        m_ldgsts_flags[i][j] = true;
        m_ldgsts_shmem_addrs[i][j] = "";
      }
    }
    // for (auto &block : (*m_insts)) {
    //   block.resize(m_n_warps_per_block);
    // }
    
  }
  void processing_finished() {
    // processing the raw file finished
    m_raw_trace_file.close();
    flush();
    combine_all_tmp_files();
  }

  void combine_all_tmp_files() {
    
    
    char *itmp_file_buf = new char[m_max_RAM / 2];

    // ofstream otrace_file;
    ifstream itmp_file;

    // otrace_file.rdbuf()->pubsetbuf(otrace_file_buf, m_max_RAM / 2);
    itmp_file.rdbuf()->pubsetbuf(itmp_file_buf, m_max_RAM / 2);

    // otrace_file.open(m_raw_trace_filepath.string() + "g");
    // if (!otrace_file.is_open()) {
    //   cerr << "Error: cannot create file " << m_raw_trace_filepath.string() + "g" << endl;
    //   exit(1);
    // }

    gzFile otrace_file = gzopen((m_raw_trace_filepath.string() + "g" + ".gz").c_str(), "wb");
    if (!otrace_file) {
      cerr << "Error: cannot create file " << m_raw_trace_filepath.string() + "g" + ".gz" << endl;
      exit(1);    
    }

    // combining the header file first
    itmp_file.open(m_kernel_dirpath / "header");
    if(!itmp_file.is_open()) 
    {
      cerr << "Error: cannot open file " << m_kernel_dirpath / "header";
      exit(1);
    } 
    
    // when the buffer becomes full we write in the file 
    const unsigned BUFF_SIZE = m_max_RAM / 2; // size of the buffer to fill before writting to file
    char *otrace_file_buf = new char[BUFF_SIZE];
    unsigned buff_pos = 0; // position of filled bytes in the buffer 

    while(!itmp_file.eof()) {
      string line;
      getline(itmp_file, line);
      line += "\n";
      unsigned n_chars_in_line = line.size(); 
      if (buff_pos + n_chars_in_line > BUFF_SIZE) {
        // buffer doesn't have enough space for read line
        // flush buffer to the file
        gzwrite(otrace_file, (voidpc)otrace_file_buf, buff_pos);
        // update buffer pos
        buff_pos = n_chars_in_line;
      }
      // copy the new line to the buffer 
      std::memcpy(otrace_file_buf + buff_pos, line.c_str(), n_chars_in_line);
      buff_pos += n_chars_in_line;
    }
    
    // if there's something left in the buffer
    if(buff_pos > 0) {
      gzwrite(otrace_file, (voidpc)otrace_file_buf, buff_pos);
      buff_pos = 0;
    }

    gzputc(otrace_file, '\n');

    itmp_file.close();
    fs::remove(m_kernel_dirpath / "header");

    // combine all the warp files and group them per block
    for (unsigned tb_id = 0; tb_id < m_insts->size(); ++tb_id) {
      assert(m_tb_initialized[tb_id]);
      std::ostringstream ss;
      ss << "\n#BEGIN_TB\n";
      ss << "\nthread block = " << get<0>(m_tb_index[tb_id]) << ',' 
                                       << get<1>(m_tb_index[tb_id]) << ',' 
                                       << get<2>(m_tb_index[tb_id]) << '\n';
      for (unsigned wid = 0; wid < (*m_insts)[tb_id].size(); ++wid) {
        unsigned gwid = tb_id * m_n_warps_per_block + wid;
        assert(m_n_tot_insts_per_warp[gwid] > 0);
        ss << "\nwarp = " << wid << '\n';
        ss << "insts = " << m_n_tot_insts_per_warp[gwid] << '\n';

        // read the insts from the tmp trace file of the warp
        fs::path warp_tmp_trace_filepath = m_kernel_dirpath / to_string(gwid);
        itmp_file.open(warp_tmp_trace_filepath);
        if (!itmp_file.is_open()) {
          cerr << "Error: cannot open the file " << warp_tmp_trace_filepath << endl;
          exit(1);
        }
        gzwrite(otrace_file, (voidpc)ss.str().c_str(), ss.str().size());
        ss.str("");
        while(!itmp_file.eof()) {
          string line;
          getline(itmp_file, line);
          if (line == "") continue;
          line += "\n";
          unsigned n_chars_in_line = line.size(); 
          if (buff_pos + n_chars_in_line > BUFF_SIZE) {
            // buffer doesn't have enough space for read line
            // flush buffer to the file
            gzwrite(otrace_file, (voidpc)otrace_file_buf, buff_pos);
            // update buffer pos
            buff_pos = n_chars_in_line;
          }
          // copy the new line to the buffer 
          std::memcpy(otrace_file_buf + buff_pos, line.c_str(), n_chars_in_line);
          buff_pos += n_chars_in_line;
        }
    
        // if there's something left in the buffer
        if(buff_pos > 0) {
          gzwrite(otrace_file, (voidpc)otrace_file_buf, buff_pos);
          buff_pos = 0;
        }
        
        itmp_file.close();
        fs::remove(warp_tmp_trace_filepath);
      }

      gzwrite(otrace_file, "\n#END_TB\n", 9);
    }
    delete [] itmp_file_buf;  
    gzclose(otrace_file); //otrace_file.close();
    delete [] otrace_file_buf;

  }

private:
  unsigned m_grid_dim_x;
  unsigned m_grid_dim_y;
  unsigned m_grid_dim_z;
  unsigned m_block_dim_x;
  unsigned m_block_dim_y;
  unsigned m_block_dim_z;
  unsigned m_n_warps_per_block; // number of warps per block
  std::vector<unsigned long long> m_n_tot_insts_per_warp; // number of instructions per warp
  std::vector<std::vector<std::vector<string>>> *m_insts; // the processed instructions indexed by (tb_id, wid, inst_id)
  std::vector<std::tuple<unsigned, unsigned, unsigned>> m_tb_index; // the indexes of the thead blocks
  std::vector<bool> m_tb_initialized; // the thread block index has captured
  unsigned long long m_tot_insts_bytes; // total size in memory to store instructions
  fs::path m_kernel_dirpath; // the path to store temprary processed files
  ifstream m_raw_trace_file; // the trace file has raw information
  unsigned long long m_max_RAM; // maximum RAM in bytes
  fs::path m_raw_trace_filepath; // the path to the trace file has raw information

  std::string m_kernel_name;
  // Add a flag for LDGSTS instruction to indicate which one to remove
  std::vector<vector<bool>> m_ldgsts_flags;  // true to remove, false to not
  std::vector<vector<std::string>> m_ldgsts_shmem_addrs;  // true to remove, false to not
  traced_execution *m_extra_trace_info;
};

class TraceProcessor {
  public:
    TraceProcessor(const std::vector<fs::path> &paths, unsigned long long max_RAM, std::string extra_trace_directory) : m_raw_trace_paths{paths}, m_max_RAM{max_RAM} {
      m_extra_trace_information.DeserializeFromFile(extra_trace_directory.c_str());
    }
    void process() {
      for (const auto& raw_kernel_trace_path : m_raw_trace_paths) {
        process_raw_kernel_trace(raw_kernel_trace_path);
      }
    }

    void process_raw_kernel_trace(const fs::path &raw_trace_filepath) {
      std::cout << "processing " << raw_trace_filepath << std::endl;
      fs::path directory = raw_trace_filepath.parent_path();
      fs::path raw_trace_fname = raw_trace_filepath.filename();
      fs::path tmp_kernel_dir_path = (directory / raw_trace_fname).replace_extension(".tmp");
  
      if (fs::exists(tmp_kernel_dir_path)) {
        std::cout << tmp_kernel_dir_path << " already exists\n";
        exit(1);
      } 
  
      if (!fs::create_directory(tmp_kernel_dir_path)) {
        cerr << "Error: unable to create directory " << tmp_kernel_dir_path << endl;
        exit(1);
      }

      Kernel kernel(tmp_kernel_dir_path, raw_trace_filepath, m_max_RAM, &m_extra_trace_information);
      kernel.process_raw_trace_file();
      assert(fs::is_directory(tmp_kernel_dir_path) && fs::is_empty(tmp_kernel_dir_path));
      fs::remove(tmp_kernel_dir_path);
    }

  private:
    const std::vector<fs::path> m_raw_trace_paths; // the path to all the raw trace files
    unsigned long long m_max_RAM;
    traced_execution m_extra_trace_information;
};

int main(int argc, char *argv[]) {

  // read the kernellist_filepath and maximum amount of RAM in GB
  fs::path kernellist_filepath;
  unsigned long long max_RAM = 0;
  if (argc == 3) {
    kernellist_filepath = argv[1];
    max_RAM = stoull(argv[2]);
  } else {
    cout << "Format: \n" 
         << "post-traces-processing kernellist_filepath max_RAM(in GB)" << endl;
    exit(1);
  }

  ifstream kernellist_file(kernellist_filepath);
  if (!kernellist_file) {
    cerr << "Error: cannot open kernellist file " << kernellist_filepath << endl;
    exit(1);
  }

  string line;
  ofstream okernellist_file(kernellist_filepath.string() + ".g");
  if (!okernellist_file) {
    cerr << "Error: unable to create " << kernellist_filepath.string() + ".g" << std::endl;
    exit(1);
  }

  vector<fs::path> raw_trace_filepaths;
  fs::path directory = kernellist_filepath.parent_path();
  while(kernellist_file >> line) {
    string command = line.substr(0, 6);
    if (command == "Memcpy"s) {
      // if it is memcpy
      okernellist_file << line << std::endl;
    } else if(command == "kernel") {
      // if it is kernel launch
      
      raw_trace_filepaths.push_back(directory / line);
      okernellist_file << line + "g" + ".gz" << std::endl;
    } else {
      cerr << "Error: Unknown command" << std::endl;
      exit(1);
    }
  }

  std::string extra_trace_information_path = directory.generic_string() + "/extra_info/enhanced_execution_info.json";


  kernellist_file.close();
  okernellist_file.close();

  TraceProcessor tp(raw_trace_filepaths, max_RAM, extra_trace_information_path);
  tp.process();

  return 0;
}