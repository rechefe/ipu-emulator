/* ETISS architecture plugin for the IPU.
 *
 * One ETISS "instruction" is one IPU VLIW word.  The word is
 * IPU_WORD_BITS wide and stored IPU_WORD_BYTES-aligned, exactly as the
 * assembler's `--format bin` output lays it out, so an assembled image is
 * executed verbatim.
 */
#ifndef IPU_ARCH_H
#define IPU_ARCH_H

#include "etiss/CPUArch.h"
#include "etiss/Instruction.h"

#include "IPU/IPU_gen.h"

/* The whole ISA is one instruction group: slot decoding happens inside the
 * single catch-all definition emitted by ipu_as.gen_etiss. */
extern etiss::instr::InstructionGroup ISA_IPU;
extern etiss::instr::InstructionClass ISA_IPUClass;
extern etiss::instr::InstructionCollection ISA_IPUCollection;

class IPUArch : public etiss::CPUArch
{
  public:
    explicit IPUArch(unsigned coreno = 0);
    virtual ~IPUArch();

    virtual ETISS_CPU *newCPU() override;
    virtual void resetCPU(ETISS_CPU *cpu, etiss::uint64 *startpointer) override;
    virtual void deleteCPU(ETISS_CPU *cpu) override;

    virtual unsigned getMaximumInstructionSizeInBytes() override;
    virtual unsigned getInstructionSizeInBytes() override;
    virtual const std::set<std::string> &getHeaders() const override;

    virtual void initInstrSet(etiss::instr::ModedInstructionSet &) const override;
    virtual void compensateEndianess(ETISS_CPU *cpu, etiss::instr::BitArray &ba) const override;
    virtual void initCodeBlock(etiss::CodeBlock &cb) const override;

    virtual std::shared_ptr<etiss::VirtualStruct> getVirtualStruct(ETISS_CPU *cpu) override;
    virtual const std::set<std::string> &getListenerSupportedRegisters() override;

  private:
    unsigned coreno_;
    std::set<std::string> headers_;
    std::set<std::string> listener_registers_;
};

#endif /* IPU_ARCH_H */
