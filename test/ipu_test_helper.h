#ifndef IPU_TEST_HELPER_H
#define IPU_TEST_HELPER_H

#ifdef __cplusplus
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <functional>

extern "C"
{
#endif
#include "ipu/ipu.h"
#include "emulator/emulator.h"
#include "fp/fp.h"
#include "xmem/xmem.h"
#include "logging/logger.h"
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace ipu_test
{

/**
 * @brief Helper class for IPU emulator tests
 * 
 * Provides utilities to:
 * - Set up IPU emulator
 * - Write inline assembly code
 * - Load programs from strings
 * - Validate results
 */
class IpuTestHelper
{
public:
    IpuTestHelper() : ipu_(nullptr), max_cycles_(10000)
    {
        ipu_ = ipu__init_ipu();
        if (!ipu_)
        {
            throw std::runtime_error("Failed to initialize IPU");
        }
    }

    ~IpuTestHelper()
    {
        if (ipu_)
        {
            if (ipu_->xmem)
            {
                free(ipu_->xmem);
            }
            free(ipu_);
        }

        // Clean up temporary files
        if (!temp_asm_file_.empty())
        {
            std::remove(temp_asm_file_.c_str());
        }
        if (!temp_bin_file_.empty())
        {
            std::remove(temp_bin_file_.c_str());
        }
    }

    /**
     * @brief Load an IPU program from inline assembly code
     * 
     * @param assembly_code Assembly code as a string
     * @return true if successful, false otherwise
     */
    bool LoadProgramFromAssembly(const std::string &assembly_code)
    {
        // Create temporary assembly file
        temp_asm_file_ = "/tmp/ipu_test_" + std::to_string(getpid()) + ".s";
        temp_bin_file_ = "/tmp/ipu_test_" + std::to_string(getpid()) + ".bin";

        // Write assembly to file
        std::ofstream asm_file(temp_asm_file_);
        if (!asm_file.is_open())
        {
            ADD_FAILURE() << "Failed to create temporary assembly file: " << temp_asm_file_;
            return false;
        }
        asm_file << assembly_code;
        asm_file.close();

        // Get assembler path from environment variable set by Bazel
        const char* ipu_assembler = std::getenv("IPU_ASSEMBLER");
        
        if (!ipu_assembler || strlen(ipu_assembler) == 0)
        {
            ADD_FAILURE() << "IPU_ASSEMBLER environment variable not set by Bazel";
            return false;
        }

        std::string assembler_cmd = ipu_assembler;

        // Assemble using ipu-as
        std::string assemble_cmd = assembler_cmd + " assemble --input " +
                                   temp_asm_file_ + " --output " + temp_bin_file_ + " --format bin";

        int ret = system(assemble_cmd.c_str());
        if (ret != 0)
        {
            ADD_FAILURE() << "Failed to assemble code. Command: " << assemble_cmd << " (ret=" << ret << ")";
            return false;
        }

        // Load binary into IPU instruction memory
        FILE *bin_file = fopen(temp_bin_file_.c_str(), "rb");
        if (!bin_file)
        {
            ADD_FAILURE() << "Failed to open assembled binary file: " << temp_bin_file_;
            return false;
        }

        ipu__load_inst_mem(ipu_, bin_file);
        fclose(bin_file);

        return true;
    }

    /**
     * @brief Load an IPU program from a binary file
     * 
     * @param bin_file_path Path to binary file
     * @return true if successful, false otherwise
     */
    bool LoadProgramFromFile(const std::string &bin_file_path)
    {
        FILE *bin_file = fopen(bin_file_path.c_str(), "rb");
        if (!bin_file)
        {
            ADD_FAILURE() << "Failed to open binary file: " << bin_file_path;
            return false;
        }

        ipu__load_inst_mem(ipu_, bin_file);
        fclose(bin_file);
        return true;
    }

    /**
     * @brief Run the IPU program until completion
     * 
     * @param max_cycles Maximum number of cycles to run (default: 10000)
     * @return Number of cycles executed, or -1 on error
     */
    int Run(uint32_t max_cycles = 0)
    {
        if (max_cycles == 0)
        {
            max_cycles = max_cycles_;
        }
        return emulator__run_until_complete(ipu_, max_cycles, max_cycles);
    }

    /**
     * @brief Execute a single instruction
     */
    void Step()
    {
        ipu__execute_next_instruction(ipu_);
    }

    /**
     * @brief Get the IPU object (for direct manipulation)
     */
    ipu__obj_t *GetIpu() { return ipu_; }

    /**
     * @brief Set maximum cycles for execution
     */
    void SetMaxCycles(uint32_t max_cycles) { max_cycles_ = max_cycles; }

    // ========== Memory Access Helpers ==========

    /**
     * @brief Write data to external memory
     */
    void WriteXmem(uint32_t addr, const std::vector<uint8_t> &data)
    {
        xmem__write_address(ipu_->xmem, addr, data.data(), data.size());
    }

    /**
     * @brief Write FP32 values to external memory as FP8_E4M3
     */
    void WriteXmemFp32AsE4M3(uint32_t addr, const std::vector<float> &values)
    {
        std::vector<uint8_t> fp8_data(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            fp__fp8_e4m3_t fp8_val = fp__fp32_to_fp8_e4m3(values[i]);
            fp8_data[i] = fp8_val.w;
        }
        xmem__write_address(ipu_->xmem, addr, fp8_data.data(), fp8_data.size());
    }

    /**
     * @brief Write FP32 values to external memory as FP8_E5M2
     */
    void WriteXmemFp32AsE5M2(uint32_t addr, const std::vector<float> &values)
    {
        std::vector<uint8_t> fp8_data(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            fp__fp8_e5m2_t fp8_val = fp__fp32_to_fp8_e5m2(values[i]);
            fp8_data[i] = fp8_val.w;
        }
        xmem__write_address(ipu_->xmem, addr, fp8_data.data(), fp8_data.size());
    }

    /**
     * @brief Read data from external memory
     */
    std::vector<uint8_t> ReadXmem(uint32_t addr, size_t size)
    {
        std::vector<uint8_t> data(size);
        xmem__read_address(ipu_->xmem, addr, data.data(), size);
        return data;
    }

    /**
     * @brief Read FP8_E4M3 values from external memory as FP32
     */
    std::vector<float> ReadXmemE4M3AsFp32(uint32_t addr, size_t count)
    {
        std::vector<uint8_t> fp8_data(count);
        xmem__read_address(ipu_->xmem, addr, fp8_data.data(), count);
        std::vector<float> values(count);
        for (size_t i = 0; i < count; i++)
        {
            fp__fp8_e4m3_t fp8_val;
            fp8_val.w = fp8_data[i];
            values[i] = fp__fp8_e4m3_to_fp32(fp8_val);
        }
        return values;
    }

    /**
     * @brief Read FP8_E5M2 values from external memory as FP32
     */
    std::vector<float> ReadXmemE5M2AsFp32(uint32_t addr, size_t count)
    {
        std::vector<uint8_t> fp8_data(count);
        xmem__read_address(ipu_->xmem, addr, fp8_data.data(), count);
        std::vector<float> values(count);
        for (size_t i = 0; i < count; i++)
        {
            fp__fp8_e5m2_t fp8_val;
            fp8_val.w = fp8_data[i];
            values[i] = fp__fp8_e5m2_to_fp32(fp8_val);
        }
        return values;
    }

    // ========== Register Access Helpers ==========

    /**
     * @brief Get value of an LR register
     */
    uint32_t GetLr(int lr_idx)
    {
        EXPECT_GE(lr_idx, 0);
        EXPECT_LT(lr_idx, IPU__LR_REGS_NUM);
        return ipu_->regfile.lr_regfile.lr[lr_idx];
    }

    /**
     * @brief Set value of an LR register
     */
    void SetLr(int lr_idx, uint32_t value)
    {
        EXPECT_GE(lr_idx, 0);
        EXPECT_LT(lr_idx, IPU__LR_REGS_NUM);
        ipu_->regfile.lr_regfile.lr[lr_idx] = value;
    }

    /**
     * @brief Get value of a CR register
     */
    uint32_t GetCr(int cr_idx)
    {
        EXPECT_GE(cr_idx, 0);
        EXPECT_LT(cr_idx, IPU__CR_REGS_NUM);
        return ipu_->regfile.cr_regfile.cr[cr_idx];
    }

    /**
     * @brief Set value of a CR register
     */
    void SetCr(int cr_idx, uint32_t value)
    {
        EXPECT_GE(cr_idx, 0);
        EXPECT_LT(cr_idx, IPU__CR_REGS_NUM);
        ipu_->regfile.cr_regfile.cr[cr_idx] = value;
    }

    /**
     * @brief Get program counter
     */
    uint32_t GetPc() { return ipu_->program_counter; }

    /**
     * @brief Set program counter
     */
    void SetPc(uint32_t pc) { ipu_->program_counter = pc; }

    /**
     * @brief Get accumulator register value at index
     */
    void SetAccWord(int word_idx, uint32_t value)
    {
        EXPECT_GE(word_idx, 0);
        EXPECT_LT(word_idx, (int)(IPU__R_ACC_REG_SIZE_BYTES / sizeof(uint32_t)));
        ipu_->regfile.acc_stage_regfile.r_acc.words[word_idx] = value;
    }

    /**
     * @brief Get accumulator register value at index
     */
    uint32_t GetAccWord(int word_idx)
    {
        EXPECT_GE(word_idx, 0);
        EXPECT_LT(word_idx, (int)(IPU__R_ACC_REG_SIZE_BYTES / sizeof(uint32_t)));
        return ipu_->regfile.acc_stage_regfile.r_acc.words[word_idx];
    }

    /**
     * @brief Get accumulator register byte at index
     */
    uint8_t GetAccByte(int byte_idx)
    {
        EXPECT_GE(byte_idx, 0);
        EXPECT_LT(byte_idx, IPU__R_ACC_REG_SIZE_BYTES);
        return ipu_->regfile.acc_stage_regfile.r_acc.bytes[byte_idx];
    }

    /**
     * @brief Reset the accumulator to zero
     */
    void ResetAcc()
    {
        for (int i = 0; i < IPU__R_ACC_REG_SIZE_BYTES; i++)
        {
            ipu_->regfile.acc_stage_regfile.r_acc.bytes[i] = 0;
        }
    }

    // ========== Multiplication Data Registers (r0, r1) ==========

    /**
     * @brief Get byte value from r0 or r1 register
     * @param reg_idx 0 for r0, 1 for r1
     * @param byte_idx Index into register (0-127)
     */
    uint8_t GetRByte(int reg_idx, int byte_idx)
    {
        EXPECT_GE(reg_idx, 0);
        EXPECT_LT(reg_idx, IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS);
        EXPECT_GE(byte_idx, 0);
        EXPECT_LT(byte_idx, IPU__R_REG_SIZE_BYTES);
        return ipu_->regfile.mult_stage_regfile.r_regs[reg_idx].bytes[byte_idx];
    }

    /**
     * @brief Get multiple bytes from r0 or r1 register as a vector
     * @param reg_idx 0 for r0, 1 for r1
     * @param offset Starting byte offset
     * @param count Number of bytes to read
     */
    std::vector<uint8_t> GetRBytes(int reg_idx, int offset, int count)
    {
        EXPECT_GE(reg_idx, 0);
        EXPECT_LT(reg_idx, IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS);
        EXPECT_GE(offset, 0);
        EXPECT_LE(offset + count, IPU__R_REG_SIZE_BYTES);
        
        std::vector<uint8_t> result(count);
        for (int i = 0; i < count; i++)
        {
            result[i] = ipu_->regfile.mult_stage_regfile.r_regs[reg_idx].bytes[offset + i];
        }
        return result;
    }

    /**
     * @brief Set byte value in r0 or r1 register
     * @param reg_idx 0 for r0, 1 for r1
     * @param byte_idx Index into register (0-127)
     * @param value Value to set
     */
    void SetRByte(int reg_idx, int byte_idx, uint8_t value)
    {
        EXPECT_GE(reg_idx, 0);
        EXPECT_LT(reg_idx, IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS);
        EXPECT_GE(byte_idx, 0);
        EXPECT_LT(byte_idx, IPU__R_REG_SIZE_BYTES);
        ipu_->regfile.mult_stage_regfile.r_regs[reg_idx].bytes[byte_idx] = value;
    }

    /**
     * @brief Set multiple bytes in r0 or r1 register from a vector
     * @param reg_idx 0 for r0, 1 for r1
     * @param offset Starting byte offset
     * @param data Vector of bytes to write
     */
    void SetRBytes(int reg_idx, int offset, const std::vector<uint8_t> &data)
    {
        EXPECT_GE(reg_idx, 0);
        EXPECT_LT(reg_idx, IPU__MULT_STAGES_REGFILE_NUM_OF_R_REGS);
        EXPECT_GE(offset, 0);
        EXPECT_LE(offset + data.size(), IPU__R_REG_SIZE_BYTES);
        
        for (size_t i = 0; i < data.size(); i++)
        {
            ipu_->regfile.mult_stage_regfile.r_regs[reg_idx].bytes[offset + i] = data[i];
        }
    }

    // ========== Cyclic Register (r_cyclic) ==========

    /**
     * @brief Get byte value from cyclic register
     * @param byte_idx Index into register (0-511)
     */
    uint8_t GetCyclicByte(int byte_idx)
    {
        EXPECT_GE(byte_idx, 0);
        EXPECT_LT(byte_idx, IPU__R_CYCLIC_REG_SIZE_BYTES);
        return ipu_->regfile.mult_stage_regfile.r_cyclic_reg.bytes[byte_idx];
    }

    /**
     * @brief Get multiple bytes from cyclic register as a vector
     * @param offset Starting byte offset
     * @param count Number of bytes to read
     */
    std::vector<uint8_t> GetCyclicBytes(int offset, int count)
    {
        EXPECT_GE(offset, 0);
        EXPECT_LT(offset, IPU__R_CYCLIC_REG_SIZE_BYTES);
        
        std::vector<uint8_t> result(count);
        for (int i = 0; i < count; i++)
        {
            uint32_t curr_idx = (offset + i) % IPU__R_CYCLIC_REG_SIZE_BYTES;
            result[i] = ipu_->regfile.mult_stage_regfile.r_cyclic_reg.bytes[curr_idx];
        }
        return result;
    }

    /**
     * @brief Set byte value in cyclic register
     * @param byte_idx Index into register (0-511)
     * @param value Value to set
     */
    void SetCyclicByte(int byte_idx, uint8_t value)
    {
        EXPECT_GE(byte_idx, 0);
        EXPECT_LT(byte_idx, IPU__R_CYCLIC_REG_SIZE_BYTES);
        ipu_->regfile.mult_stage_regfile.r_cyclic_reg.bytes[byte_idx] = value;
    }

    /**
     * @brief Set multiple bytes in cyclic register from a vector
     * @param offset Starting byte offset
     * @param data Vector of bytes to write
     */
    void SetCyclicBytes(int offset, const std::vector<uint8_t> &data)
    {
        EXPECT_GE(offset, 0);
        EXPECT_LT(offset, IPU__R_CYCLIC_REG_SIZE_BYTES);
        
        for (size_t i = 0; i < data.size(); i++)
        {
            uint32_t curr_idx = (offset + i) % IPU__R_CYCLIC_REG_SIZE_BYTES;
            ipu_->regfile.mult_stage_regfile.r_cyclic_reg.bytes[curr_idx] = data[i];
        }
    }

private:
    ipu__obj_t *ipu_;
    uint32_t max_cycles_;
    std::string temp_asm_file_;
    std::string temp_bin_file_;
};

/**
 * @brief Base test fixture for IPU tests
 * 
 * Usage:
 *   TEST_F(IpuEmulatorTest, MyTest) {
 *     helper.LoadProgramFromAssembly("set lr0 0x100;; bkpt;;");
 *     helper.Run();
 *     EXPECT_EQ(helper.GetLr(0), 0x100);
 *   }
 */
class IpuEmulatorTest : public ::testing::Test
{
protected:
    IpuTestHelper helper;
};

} // namespace ipu_test

#endif // __cplusplus

#endif // IPU_TEST_HELPER_H
