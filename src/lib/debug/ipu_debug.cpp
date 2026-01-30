#include "ipu_debug.h"

extern "C" {
#include "logging/logger.h"
#include "ipu/ipu.h"
#include "src/tools/ipu-as-py/inst_parser.h"
}

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace {

// Command handler type
using CommandHandler = std::function<bool(ipu__obj_t*, const std::vector<std::string>&)>;

// Parse a string into tokens
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Parse an integer from string (supports hex with 0x prefix)
bool parse_int(const std::string& str, int32_t& out) {
    try {
        size_t pos;
        out = std::stoi(str, &pos, 0);
        return pos == str.size();
    } catch (...) {
        return false;
    }
}

bool parse_uint(const std::string& str, uint32_t& out) {
    try {
        size_t pos;
        out = std::stoul(str, &pos, 0);
        return pos == str.size();
    } catch (...) {
        return false;
    }
}

// Print LR registers
void print_lr_registers(const ipu__obj_t* ipu) {
    std::cout << "=== LR Registers ===" << std::endl;
    for (int i = 0; i < IPU__LR_REGS_NUM; i++) {
        std::cout << "  lr" << std::setw(2) << i << " = " 
                  << std::setw(10) << ipu->regfile.lr_regfile.lr[i]
                  << " (0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << ipu->regfile.lr_regfile.lr[i] << std::dec << std::setfill(' ') << ")"
                  << std::endl;
    }
}

// Print CR registers
void print_cr_registers(const ipu__obj_t* ipu) {
    std::cout << "=== CR Registers ===" << std::endl;
    for (int i = 0; i < IPU__CR_REGS_NUM; i++) {
        std::cout << "  cr" << std::setw(2) << i << " = " 
                  << std::setw(10) << ipu->regfile.cr_regfile.cr[i]
                  << " (0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << ipu->regfile.cr_regfile.cr[i] << std::dec << std::setfill(' ') << ")"
                  << std::endl;
    }
}

// Print program counter
void print_pc(const ipu__obj_t* ipu) {
    std::cout << "=== Program Counter ===" << std::endl;
    std::cout << "  PC = " << ipu->program_counter << std::endl;
}

// Print R registers (mult stage)
void print_r_registers(const ipu__obj_t* ipu) {
    std::cout << "=== R Registers (Mult Stage) ===" << std::endl;
    for (int r = 0; r < IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS; r++) {
        std::cout << "  r" << r << " (" << IPU__R_REG_SIZE_BYTES << " bytes): ";
        for (int i = 0; i < 16; i++) {  // Print first 16 bytes
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                      << (int)ipu->regfile.mult_stage_regfile.r_regs[r].bytes[i] << " ";
        }
        std::cout << "..." << std::dec << std::setfill(' ') << std::endl;
    }
}

// Print R cyclic register
void print_rcyclic_register(const ipu__obj_t* ipu) {
    std::cout << "=== R Cyclic Register (" << IPU__R_CYCLIC_REG_SIZE_BYTES << " bytes) ===" << std::endl;
    std::cout << "  rcyclic: ";
    for (int i = 0; i < 32; i++) {  // Print first 32 bytes
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[i] << " ";
    }
    std::cout << "..." << std::dec << std::setfill(' ') << std::endl;
}

// Print R mask register
void print_rmask_register(const ipu__obj_t* ipu) {
    std::cout << "=== R Mask Register (" << IPU__R_REG_SIZE_BYTES << " bytes) ===" << std::endl;
    std::cout << "  rmask: ";
    for (int i = 0; i < 16; i++) {  // Print first 16 bytes
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)ipu->regfile.mult_stage_regfile.r_mask.bytes[i] << " ";
    }
    std::cout << "..." << std::dec << std::setfill(' ') << std::endl;
}

// Print accumulator register
void print_acc_register(const ipu__obj_t* ipu) {
    std::cout << "=== Accumulator Register (" << IPU__R_ACC_REG_SIZE_BYTES << " bytes) ===" << std::endl;
    std::cout << "  acc: ";
    for (int i = 0; i < 16; i++) {  // Print first 16 bytes
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)ipu->regfile.acc_stage_regfile.r_acc.bytes[i] << " ";
    }
    std::cout << "..." << std::dec << std::setfill(' ') << std::endl;
}

// Helper to print a range of bytes from a byte array
void print_byte_range(const uint8_t* bytes, int offset, int count, int total_size) {
    if (offset < 0 || offset >= total_size) {
        std::cerr << "Error: offset " << offset << " out of range [0, " << total_size - 1 << "]" << std::endl;
        return;
    }
    int end = std::min(offset + count, total_size);
    std::cout << "  bytes[" << offset << ".." << (end - 1) << "]: ";
    for (int i = offset; i < end; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;
}

// Helper to print words from a byte array
void print_word_range(const uint8_t* bytes, int word_offset, int count, int total_bytes) {
    int total_words = total_bytes / 4;
    if (word_offset < 0 || word_offset >= total_words) {
        std::cerr << "Error: word offset " << word_offset << " out of range [0, " << total_words - 1 << "]" << std::endl;
        return;
    }
    int end = std::min(word_offset + count, total_words);
    std::cout << "  words[" << word_offset << ".." << (end - 1) << "]: ";
    for (int i = word_offset; i < end; i++) {
        uint32_t val = *(const uint32_t*)(bytes + i * 4);
        std::cout << std::hex << std::setw(8) << std::setfill('0') << val << " ";
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;
}

// Save registers to JSON file
void save_registers_to_json(const ipu__obj_t* ipu, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    file << "{\n";
    
    // Program counter
    file << "  \"pc\": " << ipu->program_counter << ",\n";
    
    // LR registers
    file << "  \"lr\": [";
    for (int i = 0; i < IPU__LR_REGS_NUM; i++) {
        file << ipu->regfile.lr_regfile.lr[i];
        if (i < IPU__LR_REGS_NUM - 1) file << ", ";
    }
    file << "],\n";
    
    // CR registers
    file << "  \"cr\": [";
    for (int i = 0; i < IPU__CR_REGS_NUM; i++) {
        file << ipu->regfile.cr_regfile.cr[i];
        if (i < IPU__CR_REGS_NUM - 1) file << ", ";
    }
    file << "],\n";
    
    // R registers
    file << "  \"r_regs\": [\n";
    for (int r = 0; r < IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS; r++) {
        file << "    [";
        for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++) {
            file << (int)ipu->regfile.mult_stage_regfile.r_regs[r].bytes[i];
            if (i < IPU__R_REG_SIZE_BYTES - 1) file << ", ";
        }
        file << "]";
        if (r < IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";
    
    // R cyclic register
    file << "  \"r_cyclic\": [";
    for (int i = 0; i < IPU__R_CYCLIC_REG_SIZE_BYTES; i++) {
        file << (int)ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes[i];
        if (i < IPU__R_CYCLIC_REG_SIZE_BYTES - 1) file << ", ";
    }
    file << "],\n";
    
    // R mask register
    file << "  \"r_mask\": [";
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++) {
        file << (int)ipu->regfile.mult_stage_regfile.r_mask.bytes[i];
        if (i < IPU__R_REG_SIZE_BYTES - 1) file << ", ";
    }
    file << "],\n";
    
    // Accumulator
    file << "  \"acc\": [";
    for (int i = 0; i < IPU__R_ACC_REG_SIZE_BYTES; i++) {
        file << (int)ipu->regfile.acc_stage_regfile.r_acc.bytes[i];
        if (i < IPU__R_ACC_REG_SIZE_BYTES - 1) file << ", ";
    }
    file << "]\n";
    
    file << "}\n";
    file.close();
    
    std::cout << "Registers saved to " << filename << std::endl;
}

// Command: help
bool cmd_help(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)ipu;
    (void)args;
    std::cout << "Available commands:" << std::endl;
    std::cout << "  help              - Show this help message" << std::endl;
    std::cout << "  regs              - Print all registers" << std::endl;
    std::cout << "  lr                - Print LR registers" << std::endl;
    std::cout << "  cr                - Print CR registers" << std::endl;
    std::cout << "  pc                - Print program counter" << std::endl;
    std::cout << "  r                 - Print R registers (mult stage)" << std::endl;
    std::cout << "  rcyclic           - Print R cyclic register" << std::endl;
    std::cout << "  rmask             - Print R mask register" << std::endl;
    std::cout << "  acc               - Print accumulator register" << std::endl;
    std::cout << std::endl;
    std::cout << "  get lr<N>         - Get value of LR register N" << std::endl;
    std::cout << "  get cr<N>         - Get value of CR register N" << std::endl;
    std::cout << "  get r<N> [off] [cnt]  - Get bytes from R reg N (offset, count)" << std::endl;
    std::cout << "  get rcyclic [off] [cnt] - Get bytes from R cyclic (offset, count)" << std::endl;
    std::cout << "  get rmask [off] [cnt]   - Get bytes from R mask (offset, count)" << std::endl;
    std::cout << "  get acc [off] [cnt]     - Get bytes from accumulator (offset, count)" << std::endl;
    std::cout << "  getw r<N> [off] [cnt]   - Get words from R reg N" << std::endl;
    std::cout << "  getw rcyclic [off] [cnt] - Get words from R cyclic" << std::endl;
    std::cout << "  getw rmask [off] [cnt]   - Get words from R mask" << std::endl;
    std::cout << "  getw acc [off] [cnt]     - Get words from accumulator" << std::endl;
    std::cout << std::endl;
    std::cout << "  set lr<N> <val>   - Set LR register N to value" << std::endl;
    std::cout << "  set cr<N> <val>   - Set CR register N to value" << std::endl;
    std::cout << "  set pc <val>      - Set program counter to value" << std::endl;
    std::cout << "  disasm            - Disassemble current instruction" << std::endl;
    std::cout << "  save <filename>   - Save registers to JSON file" << std::endl;
    std::cout << "  step              - Execute one instruction and break again" << std::endl;
    std::cout << "  continue / c      - Continue execution" << std::endl;
    std::cout << "  quit / q          - Quit debugger and halt execution" << std::endl;
    return true;
}

// Command: regs - print all registers
bool cmd_regs(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_pc(ipu);
    print_lr_registers(ipu);
    print_cr_registers(ipu);
    print_r_registers(ipu);
    print_rcyclic_register(ipu);
    print_rmask_register(ipu);
    print_acc_register(ipu);
    return true;
}

// Command: lr - print LR registers
bool cmd_lr(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_lr_registers(ipu);
    return true;
}

// Command: cr - print CR registers
bool cmd_cr(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_cr_registers(ipu);
    return true;
}

// Command: pc - print program counter
bool cmd_pc(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_pc(ipu);
    return true;
}

// Command: r - print R registers
bool cmd_r(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_r_registers(ipu);
    return true;
}

// Command: acc - print accumulator
bool cmd_acc(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_acc_register(ipu);
    return true;
}

// Command: rcyclic - print R cyclic register
bool cmd_rcyclic(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_rcyclic_register(ipu);
    return true;
}

// Command: rmask - print R mask register
bool cmd_rmask(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    print_rmask_register(ipu);
    return true;
}

// Command: get - get register value or byte range
bool cmd_get(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: get <register> [offset] [count]" << std::endl;
        return true;
    }
    
    const std::string& reg = args[1];
    int32_t offset = 0;
    int32_t count = 16;  // Default to 16 bytes
    
    if (args.size() >= 3) {
        if (!parse_int(args[2], offset)) {
            std::cerr << "Invalid offset: " << args[2] << std::endl;
            return true;
        }
    }
    if (args.size() >= 4) {
        if (!parse_int(args[3], count)) {
            std::cerr << "Invalid count: " << args[3] << std::endl;
            return true;
        }
    }
    
    if (reg.substr(0, 2) == "lr") {
        int32_t idx;
        if (parse_int(reg.substr(2), idx) && idx >= 0 && idx < IPU__LR_REGS_NUM) {
            std::cout << "lr" << idx << " = " << ipu->regfile.lr_regfile.lr[idx] 
                      << " (0x" << std::hex << ipu->regfile.lr_regfile.lr[idx] << std::dec << ")"
                      << std::endl;
        } else {
            std::cerr << "Invalid LR register index" << std::endl;
        }
    } else if (reg.substr(0, 2) == "cr") {
        int32_t idx;
        if (parse_int(reg.substr(2), idx) && idx >= 0 && idx < IPU__CR_REGS_NUM) {
            std::cout << "cr" << idx << " = " << ipu->regfile.cr_regfile.cr[idx]
                      << " (0x" << std::hex << ipu->regfile.cr_regfile.cr[idx] << std::dec << ")"
                      << std::endl;
        } else {
            std::cerr << "Invalid CR register index" << std::endl;
        }
    } else if (reg == "pc") {
        std::cout << "pc = " << ipu->program_counter << std::endl;
    } else if (reg.size() >= 1 && reg[0] == 'r' && reg != "rcyclic" && reg != "rmask") {
        // R register: r0, r1, etc.
        int32_t idx;
        if (parse_int(reg.substr(1), idx) && idx >= 0 && idx < IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS) {
            std::cout << "r" << idx << " ";
            print_byte_range(ipu->regfile.mult_stage_regfile.r_regs[idx].bytes, offset, count, IPU__R_REG_SIZE_BYTES);
        } else {
            std::cerr << "Invalid R register index (0-" << (IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS - 1) << ")" << std::endl;
        }
    } else if (reg == "rcyclic") {
        std::cout << "rcyclic ";
        print_byte_range(ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes, offset, count, IPU__R_CYCLIC_REG_SIZE_BYTES);
    } else if (reg == "rmask") {
        std::cout << "rmask ";
        print_byte_range(ipu->regfile.mult_stage_regfile.r_mask.bytes, offset, count, IPU__R_REG_SIZE_BYTES);
    } else if (reg == "acc") {
        std::cout << "acc ";
        print_byte_range(ipu->regfile.acc_stage_regfile.r_acc.bytes, offset, count, IPU__R_ACC_REG_SIZE_BYTES);
    } else {
        std::cerr << "Unknown register: " << reg << std::endl;
    }
    
    return true;
}

// Command: getw - get register value as words
bool cmd_getw(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: getw <register> [word_offset] [count]" << std::endl;
        return true;
    }
    
    const std::string& reg = args[1];
    int32_t offset = 0;
    int32_t count = 4;  // Default to 4 words
    
    if (args.size() >= 3) {
        if (!parse_int(args[2], offset)) {
            std::cerr << "Invalid offset: " << args[2] << std::endl;
            return true;
        }
    }
    if (args.size() >= 4) {
        if (!parse_int(args[3], count)) {
            std::cerr << "Invalid count: " << args[3] << std::endl;
            return true;
        }
    }
    
    if (reg.size() >= 1 && reg[0] == 'r' && reg != "rcyclic" && reg != "rmask") {
        // R register: r0, r1, etc.
        int32_t idx;
        if (parse_int(reg.substr(1), idx) && idx >= 0 && idx < IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS) {
            std::cout << "r" << idx << " ";
            print_word_range(ipu->regfile.mult_stage_regfile.r_regs[idx].bytes, offset, count, IPU__R_REG_SIZE_BYTES);
        } else {
            std::cerr << "Invalid R register index (0-" << (IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS - 1) << ")" << std::endl;
        }
    } else if (reg == "rcyclic") {
        std::cout << "rcyclic ";
        print_word_range(ipu->regfile.mult_stage_regfile.r_cyclic_reg.bytes, offset, count, IPU__R_CYCLIC_REG_SIZE_BYTES);
    } else if (reg == "rmask") {
        std::cout << "rmask ";
        print_word_range(ipu->regfile.mult_stage_regfile.r_mask.bytes, offset, count, IPU__R_REG_SIZE_BYTES);
    } else if (reg == "acc") {
        std::cout << "acc ";
        print_word_range(ipu->regfile.acc_stage_regfile.r_acc.bytes, offset, count, IPU__R_ACC_REG_SIZE_BYTES);
    } else {
        std::cerr << "Unknown register: " << reg << " (use r0, r1, rcyclic, rmask, acc)" << std::endl;
    }
    
    return true;
}

// Command: set - set register value
bool cmd_set(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    if (args.size() < 3) {
        std::cerr << "Usage: set <register> <value>" << std::endl;
        return true;
    }
    
    const std::string& reg = args[1];
    uint32_t value;
    if (!parse_uint(args[2], value)) {
        std::cerr << "Invalid value: " << args[2] << std::endl;
        return true;
    }
    
    if (reg.substr(0, 2) == "lr") {
        int32_t idx;
        if (parse_int(reg.substr(2), idx) && idx >= 0 && idx < IPU__LR_REGS_NUM) {
            ipu->regfile.lr_regfile.lr[idx] = value;
            std::cout << "Set lr" << idx << " = " << value << std::endl;
        } else {
            std::cerr << "Invalid LR register index" << std::endl;
        }
    } else if (reg.substr(0, 2) == "cr") {
        int32_t idx;
        if (parse_int(reg.substr(2), idx) && idx >= 0 && idx < IPU__CR_REGS_NUM) {
            ipu->regfile.cr_regfile.cr[idx] = value;
            std::cout << "Set cr" << idx << " = " << value << std::endl;
        } else {
            std::cerr << "Invalid CR register index" << std::endl;
        }
    } else if (reg == "pc") {
        ipu->program_counter = value;
        std::cout << "Set pc = " << value << std::endl;
    } else {
        std::cerr << "Unknown register: " << reg << std::endl;
    }
    
    return true;
}

// Command: disasm - disassemble current instruction
bool cmd_disasm(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    (void)args;
    if (ipu->program_counter >= IPU__INST_MEM_SIZE) {
        std::cerr << "PC out of bounds" << std::endl;
        return true;
    }
    
    inst_parser__inst_t inst = ipu->inst_mem[ipu->program_counter];
    const char* disasm = inst_parser__disassemble(&inst);
    std::cout << "PC " << ipu->program_counter << ": " << disasm << std::endl;
    
    return true;
}

// Command: save - save registers to JSON
bool cmd_save(ipu__obj_t* ipu, const std::vector<std::string>& args) {
    std::string filename = "ipu_debug_dump.json";
    if (args.size() >= 2) {
        filename = args[1];
    }
    save_registers_to_json(ipu, filename);
    return true;
}

// Build command map
std::map<std::string, CommandHandler> build_command_map() {
    std::map<std::string, CommandHandler> commands;
    commands["help"] = cmd_help;
    commands["regs"] = cmd_regs;
    commands["lr"] = cmd_lr;
    commands["cr"] = cmd_cr;
    commands["pc"] = cmd_pc;
    commands["r"] = cmd_r;
    commands["rcyclic"] = cmd_rcyclic;
    commands["rmask"] = cmd_rmask;
    commands["acc"] = cmd_acc;
    commands["get"] = cmd_get;
    commands["getw"] = cmd_getw;
    commands["set"] = cmd_set;
    commands["disasm"] = cmd_disasm;
    commands["save"] = cmd_save;
    return commands;
}

} // anonymous namespace

extern "C" ipu_debug__action_t ipu_debug__enter_prompt(ipu__obj_t* ipu, ipu_debug__level_t level) {
    static auto commands = build_command_map();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "IPU Debug - Break at PC=" << ipu->program_counter << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Level 0: Print registers
    if (level >= IPU_DEBUG_LEVEL_0) {
        print_pc(ipu);
        print_lr_registers(ipu);
    }
    
    // Level 1: Print disassembled instruction
    if (level >= IPU_DEBUG_LEVEL_1) {
        if (ipu->program_counter < IPU__INST_MEM_SIZE) {
            inst_parser__inst_t inst = ipu->inst_mem[ipu->program_counter];
            const char* disasm = inst_parser__disassemble(&inst);
            std::cout << "\n=== Current Instruction ===" << std::endl;
            std::cout << "  " << disasm << std::endl;
        }
    }
    
    // Level 2: Save to JSON automatically
    if (level >= IPU_DEBUG_LEVEL_2) {
        std::string filename = "ipu_debug_pc" + std::to_string(ipu->program_counter) + ".json";
        save_registers_to_json(ipu, filename);
    }
    
    std::cout << "\nType 'help' for available commands, 'continue' or 'c' to resume execution.\n" << std::endl;
    
    // Interactive prompt loop
    std::string line;
    ipu_debug__action_t action = IPU_DEBUG_ACTION_CONTINUE;
    
    while (true) {
        std::cout << "debug >>> ";
        std::cout.flush();
        
        if (!std::getline(std::cin, line)) {
            // EOF - treat as quit
            std::cout << "\nEOF received, halting execution." << std::endl;
            ipu->program_counter = IPU__INST_MEM_SIZE;
            action = IPU_DEBUG_ACTION_QUIT;
            break;
        }
        
        auto tokens = tokenize(line);
        if (tokens.empty()) {
            continue;
        }
        
        const std::string& cmd = tokens[0];
        
        // Special commands that exit the prompt
        if (cmd == "continue" || cmd == "c") {
            std::cout << "Continuing execution..." << std::endl;
            action = IPU_DEBUG_ACTION_CONTINUE;
            break;
        }
        
        if (cmd == "quit" || cmd == "q") {
            std::cout << "Halting execution." << std::endl;
            ipu->program_counter = IPU__INST_MEM_SIZE;
            action = IPU_DEBUG_ACTION_QUIT;
            break;
        }
        
        if (cmd == "step") {
            std::cout << "Stepping one instruction..." << std::endl;
            action = IPU_DEBUG_ACTION_STEP;
            break;
        }
        
        // Look up command in map
        auto it = commands.find(cmd);
        if (it != commands.end()) {
            it->second(ipu, tokens);
        } else {
            std::cerr << "Unknown command: " << cmd << ". Type 'help' for available commands." << std::endl;
        }
    }
    
    return action;
}
