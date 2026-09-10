#include "IPUArch.h"

#include "etiss/ETISS.h"
#include "etiss/VirtualStruct.h"

#include <cstring>
#include <sstream>

/* Instruction group / class.  Mode 1 mirrors the RISC-V architectures shipped
 * with ETISS; resetCPU sets cpu->mode to the same value. */
etiss::instr::InstructionGroup ISA_IPU("ISA_IPU", IPU_WORD_FETCH_BITS);
etiss::instr::InstructionClass ISA_IPUClass(1, "ISA_IPU", IPU_WORD_FETCH_BITS, ISA_IPU);
etiss::instr::InstructionCollection ISA_IPUCollection("ISA_IPU", ISA_IPUClass);

namespace
{

/* A 32-bit scalar register (LR / CR) exposed to debuggers and the GDB stub. */
class IpuScalarField : public etiss::VirtualStruct::Field
{
  public:
    IpuScalarField(etiss::VirtualStruct &parent, const std::string &name, bool is_cr, unsigned index)
        : Field(parent, name, name, R | W, 4), is_cr_(is_cr), index_(index)
    {
    }
    virtual ~IpuScalarField() {}

  protected:
    virtual uint64_t _read(size_t) const override
    {
        IPU *c = (IPU *)parent_.structure_;
        return (uint64_t)(is_cr_ ? c->CR[index_] : c->LR[index_]);
    }
    virtual void _write(uint64_t value, size_t) override
    {
        IPU *c = (IPU *)parent_.structure_;
        if (is_cr_)
            c->CR[index_] = (etiss_uint32)value;
        else
            c->LR[index_] = (etiss_uint32)value;
    }

  private:
    bool is_cr_;
    unsigned index_;
};

/* The program counter, in *words* — the unit assembly and the Python
 * emulator use — while ETISS tracks a byte address internally. */
class IpuPcField : public etiss::VirtualStruct::Field
{
  public:
    explicit IpuPcField(etiss::VirtualStruct &parent) : Field(parent, "PC", "PC", R | W, 8) {}
    virtual ~IpuPcField() {}

  protected:
    virtual uint64_t _read(size_t) const override
    {
        IPU *c = (IPU *)parent_.structure_;
        return (uint64_t)((c->cpu.instructionPointer - IPU_IMEM_BASE) / IPU_WORD_BYTES);
    }
    virtual void _write(uint64_t value, size_t) override
    {
        IPU *c = (IPU *)parent_.structure_;
        c->cpu.instructionPointer = IPU_IMEM_BASE + value * IPU_WORD_BYTES;
        c->cpu.nextPc = c->cpu.instructionPointer;
    }
};

/* A read-only 64-bit counter (cycles and the RunStats fields). */
class IpuCounterField : public etiss::VirtualStruct::Field
{
  public:
    IpuCounterField(etiss::VirtualStruct &parent, const std::string &name, size_t offset)
        : Field(parent, name, name, R, 8), offset_(offset)
    {
    }
    virtual ~IpuCounterField() {}

  protected:
    virtual uint64_t _read(size_t) const override
    {
        const char *base = (const char *)parent_.structure_;
        etiss_uint64 v;
        std::memcpy(&v, base + offset_, sizeof(v));
        return (uint64_t)v;
    }

  private:
    size_t offset_;
};

} // namespace

IPUArch::IPUArch(unsigned coreno) : etiss::CPUArch("IPU"), coreno_(coreno)
{
    headers_.insert("IPU/IPU_gen.h");
    headers_.insert("IPU/IPUFuncs_gen.h");
    for (unsigned i = 0; i < IPU_LR_COUNT; ++i)
    {
        std::stringstream ss;
        ss << "LR" << i;
        listener_registers_.insert(ss.str());
    }
}

IPUArch::~IPUArch() {}

ETISS_CPU *IPUArch::newCPU()
{
    ETISS_CPU *ret = (ETISS_CPU *)new IPU();
    resetCPU(ret, nullptr);
    return ret;
}

/* Reset everything except instruction memory (which lives in the System, not
 * in the CPU state), matching the reset semantics of the Python harness:
 * a fresh IpuState with CR0=0, CR1=1, the default dstructure in CR15, and
 * an all-ones R_MASK. */
void IPUArch::resetCPU(ETISS_CPU *cpu, etiss::uint64 *startpointer)
{
    IPU *c = (IPU *)cpu;
    const etiss_uint32 dtype = c->dtype;
    const double elu_alpha = c->elu_alpha;
    const etiss_uint32 break_mode = c->break_mode;
    const etiss_uint64 max_cycles = c->max_cycles;

    std::memset(c, 0, sizeof(IPU));

    cpu->instructionPointer = startpointer ? *startpointer : IPU_IMEM_BASE;
    cpu->nextPc = cpu->instructionPointer;
    cpu->mode = 1;
    cpu->cpuTime_ps = 0;
    cpu->cpuCycleTime_ps = 1000;

    /* Configuration survives a reset: it is set up by the host before the
     * program runs, exactly like IpuState's constructor arguments. */
    c->dtype = dtype;
    c->elu_alpha = elu_alpha;
    c->break_mode = break_mode;
    c->max_cycles = max_cycles;

    /* RegFile defaults (ipu_emu.regfile.RegFile.__init__). */
    c->CR[0] = 0;
    c->CR[1] = 1;
    std::memset(c->R_MASK, 0xFF, sizeof(c->R_MASK));
    /* CR15 holds the default dstructure: valid_elements = LANES, P0, pad zero. */
    c->CR[IPU_CR_COUNT - 1] = IPU_LANES & 0xFF;
}

void IPUArch::deleteCPU(ETISS_CPU *cpu)
{
    delete (IPU *)cpu;
}

unsigned IPUArch::getMaximumInstructionSizeInBytes()
{
    return IPU_WORD_BYTES;
}

unsigned IPUArch::getInstructionSizeInBytes()
{
    return IPU_WORD_BYTES;
}

const std::set<std::string> &IPUArch::getHeaders() const
{
    return headers_;
}

const std::set<std::string> &IPUArch::getListenerSupportedRegisters()
{
    return listener_registers_;
}

void IPUArch::initInstrSet(etiss::instr::ModedInstructionSet &mis) const
{
    /* The JIT-compiled blocks call into libIPUFuncs.so.  ETISS always searches
     * "<jitFiles()>/etiss/jit" for libraries, which is where the plugin
     * installs it, so only the library name has to be declared. */
    std::string cfgPar = etiss::cfg().get<std::string>("jit.external_libs", ";");
    if (cfgPar.find("IPUFuncs") == std::string::npos)
        etiss::cfg().set<std::string>("jit.external_libs", cfgPar + "IPUFuncs");

    bool ok = true;
    ISA_IPUCollection.addTo(mis, ok);
    if (!ok)
        etiss::log(etiss::FATALERROR, "Failed to add the IPU instruction set");
}

/* The assembler writes each VLIW word little-endian and the fetch reads it
 * back the same way, so no compensation is needed.  The CPUArch default
 * byte-swaps in 4-byte groups, which would scramble every slot field. */
void IPUArch::compensateEndianess(ETISS_CPU *, etiss::instr::BitArray &) const {}

void IPUArch::initCodeBlock(etiss::CodeBlock &cb) const
{
    cb.fileglobalCode().insert("#include \"IPU/IPU_gen.h\"\n");
    cb.fileglobalCode().insert("#include \"IPU/IPUFuncs_gen.h\"\n");
    cb.functionglobalCode().insert("cpu->exception = 0;\n");
    cb.functionglobalCode().insert("cpu->return_pending = 0;\n");
}

std::shared_ptr<etiss::VirtualStruct> IPUArch::getVirtualStruct(ETISS_CPU *cpu)
{
    auto ret = etiss::VirtualStruct::allocate(cpu, [](etiss::VirtualStruct::Field *f) { delete f; });

    for (unsigned i = 0; i < IPU_LR_COUNT; ++i)
    {
        std::stringstream ss;
        ss << "LR" << i;
        ret->addField(new IpuScalarField(*ret, ss.str(), false, i));
    }
    for (unsigned i = 0; i < IPU_CR_COUNT; ++i)
    {
        std::stringstream ss;
        ss << "CR" << i;
        ret->addField(new IpuScalarField(*ret, ss.str(), true, i));
    }
    ret->addField(new IpuPcField(*ret));
    ret->addField(new IpuCounterField(*ret, "cycles", offsetof(IPU, cycles)));
    ret->addField(new IpuCounterField(*ret, "mult_active_cycles", offsetof(IPU, mult_active_cycles)));
    ret->addField(new IpuCounterField(*ret, "acc_active_cycles", offsetof(IPU, acc_active_cycles)));
    ret->addField(new IpuCounterField(*ret, "xmem_reads", offsetof(IPU, xmem_reads)));
    ret->addField(new IpuCounterField(*ret, "xmem_writes", offsetof(IPU, xmem_writes)));
    return ret;
}
