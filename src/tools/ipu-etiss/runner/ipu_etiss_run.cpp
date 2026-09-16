/* ipu_etiss_run — run an assembled IPU program on the ETISS backend.
 *
 * The contract with the Python driver (ipu_emu.etiss) is file-based:
 *
 *   in   --imem      raw `ipu-as assemble --format bin` image (NOP-padded)
 *   in   --state-in  register file + configuration blob (see IPU_STATE_FIELDS)
 *   in   --xmem-in   raw XMEM image, loaded at XMEM offset 0
 *   out  --state-out the same blob after the run, plus stats and error info
 *   out  --xmem-out  raw XMEM image after the run
 *
 * Exit status is 0 when the program halted normally, 2 when the IPU raised an
 * emulator error (the code is in the output blob), and 1 for usage or I/O
 * problems.
 */

#include "etiss/ETISS.h"
#include "etiss/CPUCore.h"
#include "etiss/System.h"

#include "IPU/IPU_gen.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */
/* ---------------------------------------------------------------------- */

/* Two flat regions: instruction memory (fetched from, never written by the
 * IPU) and XMEM (all data traffic).  Anything else is a bus error, which is
 * how the out-of-range XMEM accesses that raise EmulatorError in Python
 * surface here. */
class IpuSystem : public etiss::SimpleSystem
{
  public:
    IpuSystem() : imem_(IPU_INST_MEM_SIZE * IPU_WORD_BYTES, 0), xmem_(IPU_XMEM_SIZE, 0) {}

    virtual bool read(bool, ETISS_CPU *, etiss::uint64 addr, etiss::uint8 *buf, etiss::uint32 len) override
    {
        const std::vector<etiss::uint8> *src = nullptr;
        etiss::uint64 off = 0;
        if (!resolve(addr, len, &src, &off))
            return false;
        std::memcpy(buf, src->data() + off, len);
        return true;
    }

    virtual bool write(bool, ETISS_CPU *, etiss::uint64 addr, etiss::uint8 *buf, etiss::uint32 len) override
    {
        std::vector<etiss::uint8> *dst = nullptr;
        etiss::uint64 off = 0;
        if (!resolve_mut(addr, len, &dst, &off))
            return false;
        std::memcpy(dst->data() + off, buf, len);
        return true;
    }

    std::vector<etiss::uint8> imem_;
    std::vector<etiss::uint8> xmem_;

  private:
    bool resolve(etiss::uint64 addr, etiss::uint32 len, const std::vector<etiss::uint8> **region,
                 etiss::uint64 *off) const
    {
        if (addr >= IPU_IMEM_BASE && addr + len <= IPU_IMEM_BASE + imem_.size())
        {
            *region = &imem_;
            *off = addr - IPU_IMEM_BASE;
            return true;
        }
        if (addr >= IPU_XMEM_BASE && addr + len <= IPU_XMEM_BASE + xmem_.size())
        {
            *region = &xmem_;
            *off = addr - IPU_XMEM_BASE;
            return true;
        }
        return false;
    }

    bool resolve_mut(etiss::uint64 addr, etiss::uint32 len, std::vector<etiss::uint8> **region, etiss::uint64 *off)
    {
        const std::vector<etiss::uint8> *r = nullptr;
        if (!resolve(addr, len, &r, off))
            return false;
        *region = (r == &imem_) ? &imem_ : &xmem_;
        return true;
    }
};

/* ---------------------------------------------------------------------- */
/* State blob (de)serialization                                            */
/* ---------------------------------------------------------------------- */

class Blob
{
  public:
    explicit Blob(std::vector<unsigned char> data) : data_(std::move(data)) {}
    Blob() {}

    template <typename T> bool get(T *out)
    {
        if (pos_ + sizeof(T) > data_.size())
            return false;
        std::memcpy(out, data_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }
    bool get_bytes(void *out, size_t n)
    {
        if (pos_ + n > data_.size())
            return false;
        std::memcpy(out, data_.data() + pos_, n);
        pos_ += n;
        return true;
    }
    bool match(const char *magic, size_t n)
    {
        if (pos_ + n > data_.size())
            return false;
        bool ok = std::memcmp(data_.data() + pos_, magic, n) == 0;
        pos_ += n;
        return ok;
    }

    template <typename T> void put(const T &v)
    {
        const unsigned char *p = (const unsigned char *)&v;
        data_.insert(data_.end(), p, p + sizeof(T));
    }
    void put_bytes(const void *v, size_t n)
    {
        const unsigned char *p = (const unsigned char *)v;
        data_.insert(data_.end(), p, p + n);
    }

    const std::vector<unsigned char> &data() const { return data_; }

  private:
    std::vector<unsigned char> data_;
    size_t pos_ = 0;
};

bool read_file(const std::string &path, std::vector<unsigned char> *out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool write_file(const std::string &path, const void *data, size_t len)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    f.write((const char *)data, (std::streamsize)len);
    return f.good();
}

bool load_state(IPU *c, const std::vector<unsigned char> &raw)
{
    Blob b(raw);
    etiss_uint32 version = 0;
    etiss_uint64 pc = 0;
    if (!b.match(IPU_STATE_MAGIC, 8))
        return false;
    if (!b.get(&version) || version != IPU_STATE_VERSION)
        return false;
    if (!b.get(&pc))
        return false;
    c->cpu.instructionPointer = IPU_IMEM_BASE + pc * IPU_WORD_BYTES;
    c->cpu.nextPc = c->cpu.instructionPointer;

#define LOAD_U32(m, k, n) \
    if (!b.get(&c->m))    \
        return false;
#define LOAD_U64(m, k, n) LOAD_U32(m, k, n)
#define LOAD_F64(m, k, n) LOAD_U32(m, k, n)
#define LOAD_U32ARR(m, k, n)                                  \
    if (!b.get_bytes(c->m, sizeof(etiss_uint32) * (size_t)n)) \
        return false;
#define LOAD_BLOB(m, k, n)            \
    if (!b.get_bytes(c->m, (size_t)n)) \
        return false;
#define LOAD_ONE(m, k, n) LOAD_##k(m, k, n)
    IPU_STATE_FIELDS(LOAD_ONE)
#undef LOAD_ONE
#undef LOAD_BLOB
#undef LOAD_U32ARR
#undef LOAD_F64
#undef LOAD_U64
#undef LOAD_U32
    return true;
}

std::vector<unsigned char> save_state(const IPU *c)
{
    Blob b;
    b.put_bytes(IPU_STATE_MAGIC, 8);
    b.put((etiss_uint32)IPU_STATE_VERSION);
    b.put((etiss_uint64)((c->cpu.instructionPointer - IPU_IMEM_BASE) / IPU_WORD_BYTES));

#define SAVE_U32(m, k, n) b.put(c->m);
#define SAVE_U64(m, k, n) SAVE_U32(m, k, n)
#define SAVE_F64(m, k, n) SAVE_U32(m, k, n)
#define SAVE_U32ARR(m, k, n) b.put_bytes(c->m, sizeof(etiss_uint32) * (size_t)n);
#define SAVE_BLOB(m, k, n) b.put_bytes(c->m, (size_t)n);
#define SAVE_ONE(m, k, n) SAVE_##k(m, k, n)
    IPU_STATE_FIELDS(SAVE_ONE)
#undef SAVE_ONE
#undef SAVE_BLOB
#undef SAVE_U32ARR
#undef SAVE_F64
#undef SAVE_U64
#undef SAVE_U32
    return b.data();
}

void usage()
{
    std::cerr << "usage: ipu_etiss_run --imem FILE [--state-in FILE] [--xmem-in FILE]\n"
                 "                     [--state-out FILE] [--xmem-out FILE]\n"
                 "                     [--jit tcc|gcc] [--trace] [--quiet]\n";
}

} // namespace

int main(int argc, const char *argv[])
{
    std::string imem_path, state_in, state_out, xmem_in, xmem_out;
    std::string jit = "TCCJIT";
    bool trace = false;

    std::vector<const char *> etiss_argv;
    etiss_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--imem")
            imem_path = next();
        else if (a == "--state-in")
            state_in = next();
        else if (a == "--state-out")
            state_out = next();
        else if (a == "--xmem-in")
            xmem_in = next();
        else if (a == "--xmem-out")
            xmem_out = next();
        else if (a == "--jit")
            jit = next() == "gcc" ? "GCCJIT" : "TCCJIT";
        else if (a == "--trace")
            trace = true;
        else if (a == "--help" || a == "-h")
        {
            usage();
            return 0;
        }
        else
            etiss_argv.push_back(argv[i]); /* pass ETISS's own -o/-f/-i flags through */
    }

    if (imem_path.empty())
    {
        usage();
        return 1;
    }

    etiss::Initializer initializer((int)etiss_argv.size(), etiss_argv.data());
    etiss::cfg().set<std::string>("jit.type", jit);
    etiss::cfg().set<bool>("etiss.exit_on_loop", false);
    etiss::cfg().set<bool>("vp.quiet", true);

    auto sys = std::make_shared<IpuSystem>();

    std::vector<unsigned char> buf;
    if (!read_file(imem_path, &buf))
    {
        std::cerr << "ipu_etiss_run: cannot read " << imem_path << "\n";
        return 1;
    }
    if (buf.size() > sys->imem_.size())
    {
        std::cerr << "ipu_etiss_run: program image is larger than instruction memory\n";
        return 1;
    }
    std::memcpy(sys->imem_.data(), buf.data(), buf.size());

    if (!xmem_in.empty())
    {
        if (!read_file(xmem_in, &buf))
        {
            std::cerr << "ipu_etiss_run: cannot read " << xmem_in << "\n";
            return 1;
        }
        if (buf.size() > sys->xmem_.size())
        {
            std::cerr << "ipu_etiss_run: XMEM image is larger than XMEM\n";
            return 1;
        }
        std::memcpy(sys->xmem_.data(), buf.data(), buf.size());
    }

    std::shared_ptr<etiss::CPUCore> core = etiss::CPUCore::create("IPU", "core0");
    if (!core)
    {
        std::cerr << "ipu_etiss_run: failed to create the IPU core (is libIPU.so installed?)\n";
        return 1;
    }
    core->setTimer(false);
    etiss::uint64 start = IPU_IMEM_BASE;
    core->reset(&start);

    IPU *c = (IPU *)core->getState();
    if (!state_in.empty())
    {
        if (!read_file(state_in, &buf))
        {
            std::cerr << "ipu_etiss_run: cannot read " << state_in << "\n";
            return 1;
        }
        if (!load_state(c, buf))
        {
            std::cerr << "ipu_etiss_run: malformed state blob in " << state_in << "\n";
            return 1;
        }
    }

    initializer.loadIniPlugins(core);
    initializer.loadIniJIT(core);
    if (trace)
        etiss::cfg().set<int>("etiss.max_block_size", 1);

    etiss::int32 result = core->execute(*sys);

    if (!state_out.empty())
    {
        std::vector<unsigned char> out = save_state(c);
        if (!write_file(state_out, out.data(), out.size()))
        {
            std::cerr << "ipu_etiss_run: cannot write " << state_out << "\n";
            return 1;
        }
    }
    if (!xmem_out.empty() && !write_file(xmem_out, sys->xmem_.data(), sys->xmem_.size()))
    {
        std::cerr << "ipu_etiss_run: cannot write " << xmem_out << "\n";
        return 1;
    }

    if (c->error_code != 0)
        return 2;
    if (result != ETISS_RETURNCODE_CPUFINISHED && result != ETISS_RETURNCODE_NOERROR &&
        result != ETISS_RETURNCODE_BREAKPOINT)
    {
        std::cerr << "ipu_etiss_run: simulation stopped with ETISS return code " << result << "\n";
        return 1;
    }
    return 0;
}
