# IPU Emulator Test Framework

A Google Test-based framework for easily writing unit tests for IPU emulator programs.

## Features

- **Easy setup**: Single helper class handles all IPU initialization
- **Inline assembly**: Write assembly code directly in your C++ tests
- **Rich assertions**: Access registers, memory, and program state
- **Bazel integration**: Tests run with `bazel test` like any other test

## Quick Start

### Basic Test Structure

```cpp
#include "ipu_test_helper.h"

using namespace ipu_test;

TEST_F(IpuEmulatorTest, MyTest) {
    // 1. Write your assembly code
    std::string asm_code = R"(
set lr0 0x1000;;
incr lr0 5;;
bkpt;;
)";

    // 2. Load and run the program
    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();

    // 3. Validate results
    EXPECT_EQ(helper.GetLr(0), 0x1005);
}
```

## API Reference

### IpuTestHelper Class

The main helper class providing all functionality for IPU testing.

#### Program Loading

```cpp
// Load from inline assembly string
bool LoadProgramFromAssembly(const std::string &assembly_code);

// Load from pre-assembled binary file
bool LoadProgramFromFile(const std::string &bin_file_path);
```

#### Program Execution

```cpp
// Run until completion (or max cycles)
int Run(uint32_t max_cycles = 10000);

// Execute a single instruction
void Step();

// Set maximum cycles for execution
void SetMaxCycles(uint32_t max_cycles);
```

#### Register Access

```cpp
// LR registers
uint32_t GetLr(int lr_idx);
void SetLr(int lr_idx, uint32_t value);

// CR registers
uint32_t GetCr(int cr_idx);
void SetCr(int cr_idx, uint32_t value);

// Program counter
uint32_t GetPc();
void SetPc(uint32_t pc);

// Accumulator
uint32_t GetAccWord(int word_idx);
uint8_t GetAccByte(int byte_idx);
void ResetAcc();
```

#### Memory Access

```cpp
// Raw byte access
void WriteXmem(uint32_t addr, const std::vector<uint8_t> &data);
std::vector<uint8_t> ReadXmem(uint32_t addr, size_t size);

// FP8 E4M3 format
void WriteXmemFp32AsE4M3(uint32_t addr, const std::vector<float> &values);
std::vector<float> ReadXmemE4M3AsFp32(uint32_t addr, size_t count);

// FP8 E5M2 format
void WriteXmemFp32AsE5M2(uint32_t addr, const std::vector<float> &values);
std::vector<float> ReadXmemE5M2AsFp32(uint32_t addr, size_t count);
```

#### Direct IPU Access

```cpp
// Get raw IPU object for advanced operations
ipu__obj_t *GetIpu();
```

## Example Tests

### Test 1: Simple Register Operations

```cpp
TEST_F(IpuEmulatorTest, RegisterArithmetic) {
    std::string asm_code = R"(
set lr0 10;;
incr lr0 5;;
incr lr0 3;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();
    EXPECT_EQ(helper.GetLr(0), 18);
}
```

### Test 2: Memory Operations

```cpp
TEST_F(IpuEmulatorTest, MemoryReadWrite) {
    // Write test data
    std::vector<uint8_t> data(128, 0xAB);
    helper.WriteXmem(0x1000, data);

    std::string asm_code = R"(
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();
    
    // Verify data was loaded
    EXPECT_EQ(helper.GetLr(0), 0x1000);
}
```

### Test 3: Conditional Branching

```cpp
TEST_F(IpuEmulatorTest, BranchOnEqual) {
    std::string asm_code = R"(
set lr0 10;;
set lr1 10;;
beq lr0 lr1 equal_case;;
set lr2 0;;
bkpt;;
equal_case:
set lr2 1;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run();
    EXPECT_EQ(helper.GetLr(2), 1);
}
```

### Test 4: Loops

```cpp
TEST_F(IpuEmulatorTest, CountingLoop) {
    std::string asm_code = R"(
set lr0 0;;      # Counter
set lr1 100;;    # Target
loop:
incr lr0 1;;
bne lr0 lr1 loop;;
bkpt;;
)";

    ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code));
    helper.Run(5000);  // Allow enough cycles
    EXPECT_EQ(helper.GetLr(0), 100);
}
```

### Test 5: FP8 Data Processing

```cpp
TEST_F(IpuEmulatorTest, FloatingPointData) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    helper.WriteXmemFp32AsE4M3(0x2000, input);
    
    auto output = helper.ReadXmemE4M3AsFp32(0x2000, input.size());
    
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(output[i], input[i], 0.1f);
    }
}
```

## Running Tests

### Build and run all tests:
```bash
bazel test //:ipu_emulator_tests
```

### Run with verbose output:
```bash
bazel test //:ipu_emulator_tests --test_output=all
```

### Run specific test:
```bash
bazel test //:ipu_emulator_tests --test_filter=IpuEmulatorTest.RegisterArithmetic
```

### Run in debug mode:
```bash
bazel test //:ipu_emulator_tests -c dbg
```

## Adding New Tests

1. **Create your test file** (or add to existing):

```cpp
#include "ipu_test_helper.h"

using namespace ipu_test;

TEST_F(IpuEmulatorTest, YourNewTest) {
    // Your test code here
}
```

2. **If creating a new file**, add it to BUILD.bazel:

```python
cc_test(
    name = "your_test_suite",
    size = "medium",
    srcs = [
        "test/your_test.cpp",
        "test/ipu_test_helper.h",
    ],
    data = [
        "//src/tools/ipu-as-py:ipu-as",
    ],
    deps = [
        ":emulator",
        ":fp",
        ":ipu",
        ":logger",
        ":xmem",
        "@googletest//:gtest_main",
    ],
)
```

3. **Run your test**:
```bash
bazel test //:your_test_suite
```

## Best Practices

### 1. Keep Tests Focused
Each test should verify one specific behavior:
```cpp
// Good: Tests one thing
TEST_F(IpuEmulatorTest, IncrementWorks) {
    // ... test increment only
}

// Bad: Tests too many things
TEST_F(IpuEmulatorTest, EverythingWorks) {
    // ... tests increment, load, store, branches, etc.
}
```

### 2. Use Descriptive Names
```cpp
// Good
TEST_F(IpuEmulatorTest, BranchOnEqualJumpsWhenRegistersMatch)

// Bad
TEST_F(IpuEmulatorTest, Test1)
```

### 3. Set Appropriate Max Cycles
```cpp
// For simple tests
helper.Run(100);

// For loops/complex operations
helper.Run(10000);

// Or set globally
helper.SetMaxCycles(5000);
```

### 4. Clean Test Data
```cpp
TEST_F(IpuEmulatorTest, DataTest) {
    // Initialize memory with known values
    std::vector<uint8_t> data(128, 0);
    helper.WriteXmem(0x1000, data);
    
    // ... rest of test
}
```

### 5. Check Assembly Errors
```cpp
ASSERT_TRUE(helper.LoadProgramFromAssembly(asm_code))
    << "Failed to load assembly";
```

## Debugging Tips

### 1. Enable Logging
The IPU library uses a logging system. Check the output for debug info.

### 2. Step Through Instructions
```cpp
helper.Step();  // Execute one instruction
EXPECT_EQ(helper.GetPc(), 1);  // Check PC advanced

helper.Step();  // Next instruction
// ... check state
```

### 3. Check Register State
```cpp
// Print all LR registers
for (int i = 0; i < 16; i++) {
    std::cout << "LR[" << i << "] = " << helper.GetLr(i) << std::endl;
}
```

### 4. Verify Program Loaded
```cpp
auto *ipu = helper.GetIpu();
// Check first instruction is not zero
ASSERT_NE(ipu->inst_mem[0].instruction, 0);
```

## Limitations

1. **Assembler dependency**: Tests require the `ipu-as` assembler to be built first
2. **Temporary files**: Assembly is written to `/tmp` - ensure write permissions
3. **Performance**: Inline assembly has overhead from assembling on each test
4. **Platform**: Currently uses system() calls, which are platform-dependent

## Advanced Usage

### Direct IPU Manipulation
```cpp
TEST_F(IpuEmulatorTest, DirectAccess) {
    ipu__obj_t *ipu = helper.GetIpu();
    
    // Directly manipulate IPU state
    ipu->regfile.lr_regfile.lr[0] = 0x5000;
    ipu->program_counter = 10;
    
    // Continue execution from modified state
    helper.Run();
}
```

### Pre-assembled Programs
For frequently used programs, assemble once and load the binary:
```cpp
// Assemble outside the test
// bazel run //src/tools/ipu-as-py:ipu-as -- assemble \
//   --input program.s --output program.bin --format bin

TEST_F(IpuEmulatorTest, PreAssembled) {
    ASSERT_TRUE(helper.LoadProgramFromFile("testdata/program.bin"));
    helper.Run();
}
```

## Troubleshooting

### "Failed to assemble code"
- Check assembly syntax
- Ensure `ipu-as` is built: `bazel build //src/tools/ipu-as-py:ipu-as`
- Verify temporary directory is writable

### "Maximum cycle limit reached"
- Increase max cycles: `helper.SetMaxCycles(100000);`
- Check for infinite loops in your code
- Ensure your program has `bkpt;;` to stop execution

### "Failed to initialize IPU"
- Check memory allocation
- Verify all dependencies are linked correctly

## Contributing

When adding new helper functions to `ipu_test_helper.h`:
1. Document with clear comments
2. Add example usage in this README
3. Include tests for the helper function itself
4. Keep the API simple and consistent
