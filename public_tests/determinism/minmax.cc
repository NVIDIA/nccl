// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum Value {
  PZero,
  NZero,
  One,
  NegTwo,
  Poison,
  PInf,
  NInf,
  Max,
  NegMax,
  Normal,
  NegNormal,
  Qa,
  Qb,
  AnyNaN
};
struct Format {
  const char* name;
  ncclDataType_t datatype;
  int bytes;
  uint32_t width, sign, exponent, fraction;
  uint32_t values[13];
};
const Format formats[] = {
  {"fp32",
   ncclFloat32,
   4,
   0xffffffffu,
   0x80000000u,
   0x7f800000u,
   0x007fffffu,
   {0, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x40400000u, 0x7f800000u, 0xff800000u, 0x7f7fffffu, 0xff7fffffu,
    0x00800000u, 0x80800000u, 0x7fc00001u, 0xffc12345u}},
  {"fp16",
   ncclFloat16,
   2,
   0xffff,
   0x8000,
   0x7c00,
   0x03ff,
   {0, 0x8000, 0x3c00, 0xc000, 0x4200, 0x7c00, 0xfc00, 0x7bff, 0xfbff, 0x0400, 0x8400, 0x7e01, 0xfe55}},
  {"bf16",
   ncclBfloat16,
   2,
   0xffff,
   0x8000,
   0x7f80,
   0x007f,
   {0, 0x8000, 0x3f80, 0xc000, 0x4040, 0x7f80, 0xff80, 0x7f7f, 0xff7f, 0x0080, 0x8080, 0x7fc1, 0xffe5}}
};
struct Pair {
  Value a, b, minimum, maximum;
};
const Pair pairs[] = {
  {NegTwo, One, NegTwo, One},   {One, One, One, One},         {Qa, One, One, One},
  {One, Qb, One, One},          {Qa, NegTwo, NegTwo, NegTwo}, {Qa, Qb, AnyNaN, AnyNaN},
  {NZero, PZero, NZero, PZero}, {PZero, NZero, NZero, PZero}, {NZero, NZero, NZero, NZero},
  {PZero, PZero, PZero, PZero}, {NInf, PInf, NInf, PInf},     {Qa, PInf, PInf, PInf},
  {Qb, NInf, NInf, NInf},       {NegMax, Max, NegMax, Max},   {NegNormal, Normal, NegNormal, Normal},
  {Qa, NZero, NZero, NZero},    {Qb, PZero, PZero, PZero}
};
constexpr int pairCount = sizeof(pairs) / sizeof(pairs[0]);

bool isNaN(const Format& f, uint32_t bits) {
  return (bits & f.exponent) == f.exponent && (bits & f.fraction) != 0;
}
bool isZero(const Format& f, uint32_t bits) {
  return (bits & (f.width ^ f.sign)) == 0;
}
uint32_t orderKey(const Format& f, uint32_t bits) {
  return (bits & f.sign) ? (~bits & f.width) : (bits ^ f.sign);
}
enum class Kind {
  Exact,
  NaN,
  Zero
};
struct Expected {
  Kind kind;
  uint32_t bits;
};

// Order raw encodings after excluding NaNs; no host or device floating-point operations.
Expected oracle(const Format& f, bool maximum, const std::vector<uint32_t>& inputs) {
  bool found = false;
  uint32_t selected = 0;
  for (uint32_t bits : inputs) {
    if (isNaN(f, bits)) continue;
    if (!found || (maximum ? orderKey(f, bits) > orderKey(f, selected) : orderKey(f, bits) < orderKey(f, selected)))
      selected = bits;
    found = true;
  }
  if (!found) return {Kind::NaN, 0};
  if (f.datatype == ncclFloat32 && isZero(f, selected)) return {Kind::Zero, 0};
  return {Kind::Exact, selected};
}
bool accepts(const Format& f, Expected expected, uint32_t actual) {
  if (expected.kind == Kind::NaN) return isNaN(f, actual);
  if (expected.kind == Kind::Zero) return isZero(f, actual);
  return expected.bits == actual;
}
Expected tableExpected(const Format& f, Value value) {
  if (value == AnyNaN) return {Kind::NaN, 0};
  if (f.datatype == ncclFloat32 && (value == PZero || value == NZero)) return {Kind::Zero, 0};
  return {Kind::Exact, f.values[value]};
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

uint32_t inputBits(const Format& f, int row, int rank, int nranks, int sparseShift) {
  const Pair& p = pairs[row % pairCount];
  if (sparseShift < 0) return f.values[(rank % 2) ? p.b : p.a];
  int position = (rank + nranks - sparseShift) % nranks;
  return f.values[position == 0 ? p.a : position == 1 ? p.b : (position % 2) ? Qb : Qa];
}
int inputRow(int index, int count, int rotation, bool scatter) {
  return (scatter ? index % count + 3 * (index / count) : index) + rotation;
}

void selfTest() {
  for (const Format& f : formats) {
    require(isNaN(f, f.values[Qa]) && isNaN(f, f.values[Qb]) && !isNaN(f, f.values[PInf]) && !isNaN(f, f.values[NInf]),
            "NaN classifier self-test");
    const Value ordered[] = {NInf, NegMax, NegTwo, NegNormal, NZero, PZero, Normal, One, Poison, Max, PInf};
    for (size_t i = 1; i < sizeof(ordered) / sizeof(ordered[0]); ++i)
      require(orderKey(f, f.values[ordered[i - 1]]) < orderKey(f, f.values[ordered[i]]), "bit order self-test");
    for (bool maximum : {false, true}) {
      for (int nranks : {2, 6}) {
        for (int shift = -1; shift < (nranks == 6 ? 6 : 0); ++shift) {
          for (int row = 0; row < pairCount; ++row) {
            std::vector<uint32_t> inputs;
            for (int rank = 0; rank < nranks; ++rank) inputs.push_back(inputBits(f, row, rank, nranks, shift));
            Expected got = oracle(f, maximum, inputs);
            Expected expected = tableExpected(f, maximum ? pairs[row].maximum : pairs[row].minimum);
            require(got.kind == expected.kind && (got.kind != Kind::Exact || got.bits == expected.bits),
                    "oracle/table self-test");
            require(!accepts(f, got, f.values[Poison]), "poison rejection self-test");
            if (got.kind == Kind::NaN) {
              require(accepts(f, got, f.values[Qa]) && accepts(f, got, f.values[Qb]) &&
                        !accepts(f, got, f.values[PInf]),
                      "all-NaN checker self-test");
            } else {
              require(!accepts(f, got, f.values[Qa]), "numeric/NaN checker self-test");
              if (got.kind == Kind::Zero)
                require(accepts(f, got, f.values[PZero]) && accepts(f, got, f.values[NZero]), "fp32 zero self-test");
              else if (isZero(f, got.bits))
                require(!accepts(f, got, got.bits ^ f.sign), "signed zero rejection self-test");
            }
          }
        }
        // Check the generated data, so accidentally repeating destination slices fails this control.
        std::vector<std::vector<uint32_t>> snapshots(nranks, std::vector<uint32_t>(nranks * pairCount));
        for (int source = 0; source < nranks; ++source)
          for (int i = 0; i < nranks * pairCount; ++i)
            snapshots[source][i] = inputBits(f, inputRow(i, pairCount, 0, true), source, nranks, -1);
        for (int dest = 0; dest < nranks; ++dest) {
          for (int wrong = 0; wrong < nranks; ++wrong) {
            if (dest == wrong) continue;
            bool rejected = false;
            for (int i = 0; i < pairCount; ++i) {
              std::vector<uint32_t> wantedInputs, otherInputs;
              for (int source = 0; source < nranks; ++source) {
                wantedInputs.push_back(snapshots[source][dest * pairCount + i]);
                otherInputs.push_back(snapshots[source][wrong * pairCount + i]);
              }
              Expected expected = oracle(f, maximum, wantedInputs);
              Expected other = oracle(f, maximum, otherInputs);
              uint32_t bad = other.kind == Kind::NaN  ? f.values[Qa] :
                             other.kind == Kind::Zero ? f.values[PZero] :
                                                        other.bits;
              rejected |= !accepts(f, expected, bad);
            }
            require(rejected, "ReduceScatter wrong-slice rejection self-test");
          }
        }
      }
    }
  }
}

void cudaCheck(cudaError_t result, const char* call) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
void ncclCheck(ncclResult_t result, const char* call) {
  if (result != ncclSuccess) throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(result));
}
#define CUDA(call) cudaCheck((call), #call)
#define NCCL(call) ncclCheck((call), #call)

enum class Collective {
  AllReduce,
  Reduce,
  ReduceScatter
};
struct Options {
  int gpus = 2, root = -1, limit = 0;
  Collective collective = Collective::AllReduce;
  std::string collectiveName = "allreduce", algo = "Ring", proto = "Simple";
  bool selfTestOnly = false, injectError = false;
};
int integer(const std::string& text) {
  size_t end = 0;
  int value = std::stoi(text, &end);
  require(end == text.size() && value >= 0, "expected a nonnegative integer: " + text);
  return value;
}
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--self-test") {
      o.selfTestOnly = true;
      continue;
    }
    if (arg == "--inject-error") {
      o.injectError = true;
      continue;
    }
    if (arg == "--help") {
      std::cout << "minmax [--gpus 2|6] [--collective allreduce|reduce|reducescatter]\n"
                   "       [--algo Ring|Tree] [--proto Simple|LL] [--root N] [--case-limit N]\n"
                   "       [--self-test] [--inject-error]\n";
      std::exit(0);
    }
    require(i + 1 < argc, "missing value for " + arg);
    std::string value = argv[++i];
    if (arg == "--gpus") o.gpus = integer(value);
    else if (arg == "--root") o.root = integer(value);
    else if (arg == "--case-limit") {
      o.limit = integer(value);
      require(o.limit > 0, "case limit must be positive");
    } else if (arg == "--algo") o.algo = value;
    else if (arg == "--proto") o.proto = value;
    else if (arg == "--collective") {
      o.collectiveName = value;
      if (value == "allreduce") o.collective = Collective::AllReduce;
      else if (value == "reduce") o.collective = Collective::Reduce;
      else if (value == "reducescatter") o.collective = Collective::ReduceScatter;
      else throw std::runtime_error("unknown collective: " + value);
    } else throw std::runtime_error("unknown option: " + arg);
  }
  require(o.gpus == 2 || o.gpus == 6, "--gpus must be 2 or 6");
  require(o.algo == "Ring" || (o.algo == "Tree" && o.collective == Collective::AllReduce),
          "Ring supports all three collectives; Tree requires allreduce");
  require(o.proto == "Simple" || o.proto == "LL", "--proto must be Simple or LL");
  require(o.root < o.gpus && (o.root == -1 || o.collective == Collective::Reduce),
          "--root requires reduce and a valid rank");
  require(!o.selfTestOnly || !o.injectError, "--inject-error requires a GPU run");
  return o;
}
void setEnvironment(const char* name, const char* value) {
  require(setenv(name, value, 1) == 0, std::string("setenv failed: ") + name);
  std::cout << "PROFILE " << name << '=' << value << '\n';
}
void configure(const Options& o) {
  setEnvironment("NCCL_ALGO", o.algo.c_str());
  setEnvironment("NCCL_PROTO", o.proto.c_str());
  setEnvironment("NCCL_SYM_KERNEL", "");
  setEnvironment("NCCL_NVLS_ENABLE", "0");
  setEnvironment("NCCL_COLLNET_ENABLE", "0");
  setEnvironment("NCCL_TUNER_PLUGIN", "none");
  setEnvironment("NCCL_CONF_FILE", "/dev/null");
  setEnvironment("NCCL_ENQUEUE_REARCH_ENABLE", "1");
}

constexpr size_t guardBytes = 64;
struct Buffer {
  unsigned char* device = nullptr;
  std::vector<unsigned char> before, after;
  size_t payloadBytes = 0;
  unsigned char* payload() {
    return device + guardBytes;
  }
  void allocate(size_t capacity) {
    before.resize(capacity + 2 * guardBytes);
    after.resize(before.size());
    CUDA(cudaMalloc(reinterpret_cast<void**>(&device), before.size()));
  }
  void reset(size_t bytes, unsigned char guard) {
    payloadBytes = bytes;
    std::fill(before.begin(), before.end(), guard);
  }
  void upload(cudaStream_t stream) {
    CUDA(cudaMemcpyAsync(device, before.data(), before.size(), cudaMemcpyHostToDevice, stream));
  }
  void download(cudaStream_t stream) {
    CUDA(cudaMemcpyAsync(after.data(), device, after.size(), cudaMemcpyDeviceToHost, stream));
  }
  void checkGuards(const std::string& label) const {
    for (size_t i = 0; i < before.size(); ++i)
      if (i < guardBytes || i >= guardBytes + payloadBytes)
        require(after[i] == before[i], label + " guard mismatch at byte " + std::to_string(i));
  }
};
uint32_t load(const std::vector<unsigned char>& data, size_t element, int bytes) {
  uint32_t bits = 0;
  std::memcpy(&bits, data.data() + guardBytes + element * bytes, bytes);
  return bits;
}
void put(std::vector<unsigned char>& data, size_t element, int bytes, uint32_t bits) {
  std::memcpy(data.data() + guardBytes + element * bytes, &bits, bytes);
}

struct Devices {
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<Buffer> inputs, outputs;
  explicit Devices(int count) : comms(count, nullptr), streams(count, nullptr), inputs(count), outputs(count) {}
  ~Devices() {
    for (size_t rank = 0; rank < comms.size(); ++rank) {
      cudaSetDevice(static_cast<int>(rank));
      if (comms[rank]) ncclCommAbort(comms[rank]);
      if (inputs[rank].device) cudaFree(inputs[rank].device);
      if (outputs[rank].device) cudaFree(outputs[rank].device);
      if (streams[rank]) cudaStreamDestroy(streams[rank]);
    }
  }
  void initialize() {
    for (size_t rank = 0; rank < comms.size(); ++rank) {
      CUDA(cudaSetDevice(static_cast<int>(rank)));
      CUDA(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
      inputs[rank].allocate(17 * comms.size() * sizeof(uint32_t));
      outputs[rank].allocate(17 * sizeof(uint32_t));
    }
    NCCL(ncclCommInitAll(comms.data(), static_cast<int>(comms.size()), nullptr));
  }
  void synchronize() {
    for (size_t rank = 0; rank < comms.size(); ++rank) {
      CUDA(cudaSetDevice(static_cast<int>(rank)));
      CUDA(cudaStreamSynchronize(streams[rank]));
    }
  }
  void close() {
    for (size_t rank = 0; rank < comms.size(); ++rank) {
      CUDA(cudaSetDevice(static_cast<int>(rank)));
      NCCL(ncclCommDestroy(comms[rank]));
      comms[rank] = nullptr;
      CUDA(cudaFree(inputs[rank].device));
      inputs[rank].device = nullptr;
      CUDA(cudaFree(outputs[rank].device));
      outputs[rank].device = nullptr;
      CUDA(cudaStreamDestroy(streams[rank]));
      streams[rank] = nullptr;
    }
  }
};

struct Case {
  int format;
  bool maximum, inplace;
  int count, rotation, sparseShift;
};
std::vector<Case> cases(int nranks) {
  std::vector<Case> result;
  for (int format = 0; format < 3; ++format)
    for (bool maximum : {false, true})
      for (bool inplace : {false, true}) {
        for (int count : {1, 3, 4, 17})
          for (int rotation = 0; rotation < (count == 17 ? 2 : pairCount); ++rotation)
            result.push_back({format, maximum, inplace, count, rotation, -1});
        if (nranks == 6)
          for (int shift = 0; shift < nranks; ++shift) result.push_back({format, maximum, inplace, 17, 0, shift});
      }
  return result;
}
std::string hex(uint32_t bits, const Format& f) {
  std::ostringstream text;
  text << "0x" << std::hex << std::setfill('0') << std::setw(2 * f.bytes) << bits;
  return text.str();
}

void runCase(Devices& d, const Options& o, const Case& c, int caseId, bool corrupt) {
  const Format& f = formats[c.format];
  bool scatter = o.collective == Collective::ReduceScatter;
  int sendCount = scatter ? c.count * o.gpus : c.count;
  int root = o.root >= 0 ? o.root : caseId % o.gpus;
  std::ostringstream description;
  description << "case=" << caseId << " collective=" << o.collectiveName << " type=" << f.name
              << " op=" << (c.maximum ? "max" : "min") << " inplace=" << c.inplace << " count=" << c.count
              << " rotation=" << c.rotation << " sparse_shift=" << c.sparseShift;
  if (o.collective == Collective::Reduce) description << " root=" << root;
  const std::string label = description.str();
  for (int rank = 0; rank < o.gpus; ++rank) {
    CUDA(cudaSetDevice(rank));
    d.inputs[rank].reset(sendCount * f.bytes, 0xa5);
    d.outputs[rank].reset(c.count * f.bytes, 0x5a);
    for (int i = 0; i < sendCount; ++i) {
      int row = inputRow(i, c.count, c.rotation, scatter);
      put(d.inputs[rank].before, i, f.bytes, inputBits(f, row, rank, o.gpus, c.sparseShift));
    }
    for (int i = 0; i < c.count; ++i) put(d.outputs[rank].before, i, f.bytes, f.values[Poison]);
    d.inputs[rank].upload(d.streams[rank]);
    d.outputs[rank].upload(d.streams[rank]);
  }
  auto output = [&](int rank) -> unsigned char* {
    if (!c.inplace || (o.collective == Collective::Reduce && rank != root)) return d.outputs[rank].payload();
    return d.inputs[rank].payload() + (scatter ? rank * c.count * f.bytes : 0);
  };
  NCCL(ncclGroupStart());
  try {
    for (int rank = 0; rank < o.gpus; ++rank) {
      CUDA(cudaSetDevice(rank));
      ncclRedOp_t op = c.maximum ? ncclMax : ncclMin;
      if (o.collective == Collective::AllReduce)
        NCCL(ncclAllReduce(d.inputs[rank].payload(), output(rank), c.count, f.datatype, op, d.comms[rank],
                           d.streams[rank]));
      else if (o.collective == Collective::Reduce)
        NCCL(ncclReduce(d.inputs[rank].payload(), output(rank), c.count, f.datatype, op, root, d.comms[rank],
                        d.streams[rank]));
      else
        NCCL(ncclReduceScatter(d.inputs[rank].payload(), output(rank), c.count, f.datatype, op, d.comms[rank],
                               d.streams[rank]));
    }
  } catch (...) {
    ncclGroupEnd();
    throw;
  }
  NCCL(ncclGroupEnd());
  if (corrupt) {
    d.synchronize();
    int rank = o.collective == Collective::Reduce ? root : o.gpus - 1;
    CUDA(cudaSetDevice(rank));
    CUDA(cudaMemcpy(output(rank) + (c.count - 1) * f.bytes, &f.values[Poison], f.bytes, cudaMemcpyHostToDevice));
    std::cout << "INJECTED_ERROR " << label << " rank=" << rank << " element=" << c.count - 1 << '\n';
  }
  for (int rank = 0; rank < o.gpus; ++rank) {
    CUDA(cudaSetDevice(rank));
    d.inputs[rank].download(d.streams[rank]);
    d.outputs[rank].download(d.streams[rank]);
  }
  d.synchronize();
  for (int rank = 0; rank < o.gpus; ++rank) {
    std::string rankLabel = label + " rank=" + std::to_string(rank);
    d.inputs[rank].checkGuards(rankLabel + " input");
    d.outputs[rank].checkGuards(rankLabel + " output");
    if (!c.inplace || (o.collective == Collective::Reduce && rank != root))
      require(d.inputs[rank].before == d.inputs[rank].after, rankLabel + " input modified");
    if (o.collective == Collective::Reduce && rank != root) continue;
    for (int i = 0; i < c.count; ++i) {
      int sourceIndex = scatter ? rank * c.count + i : i;
      std::vector<uint32_t> operands;
      for (int source = 0; source < o.gpus; ++source)
        operands.push_back(load(d.inputs[source].before, sourceIndex, f.bytes));
      Expected expected = oracle(f, c.maximum, operands);
      uint32_t actual =
        load(c.inplace ? d.inputs[rank].after : d.outputs[rank].after, c.inplace ? sourceIndex : i, f.bytes);
      if (!accepts(f, expected, actual)) {
        std::ostringstream error;
        error << rankLabel << " element=" << i << " expected="
              << (expected.kind == Kind::NaN  ? "AnyNaN" :
                  expected.kind == Kind::Zero ? "AnyZero" :
                                                hex(expected.bits, f))
              << " actual=" << hex(actual, f) << " inputs=";
        for (uint32_t operand : operands) error << hex(operand, f) << ',';
        throw std::runtime_error(error.str());
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    Options o = parse(argc, argv);
    const uint32_t endian = 1;
    require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "raw input encoding requires a little-endian host");
    selfTest();
    if (o.selfTestOnly) {
      std::cout << "CHECKER_SELF_TEST PASS\n";
      return 0;
    }
    configure(o);
    int available = 0;
    CUDA(cudaGetDeviceCount(&available));
    if (available < o.gpus) {
      std::cerr << "UNAVAILABLE requested=" << o.gpus << " visible=" << available << '\n';
      return 77;
    }
    for (int rank = 0; rank < o.gpus; ++rank) {
      cudaDeviceProp prop{};
      CUDA(cudaGetDeviceProperties(&prop, rank));
      std::cout << "DEVICE rank=" << rank << " name=" << prop.name << " sm=" << prop.major << prop.minor << '\n';
      if (prop.major < 8) {
        std::cerr << "UNAVAILABLE requires SM80 or newer for FP16/BF16 GPU arithmetic\n";
        return 77;
      }
    }
    int runtime = 0, driver = 0, nccl = 0;
    CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaDriverGetVersion(&driver));
    NCCL(ncclGetVersion(&nccl));
    Dl_info library{};
    require(dladdr(reinterpret_cast<void*>(ncclAllReduce), &library) && library.dli_fname,
            "cannot identify loaded NCCL library");
    std::cout << "VERSIONS runtime=" << runtime << " driver=" << driver << " nccl=" << nccl
              << " library=" << library.dli_fname << '\n';
    auto matrix = cases(o.gpus);
    if (o.limit > 0) {
      require(o.limit <= static_cast<int>(matrix.size()), "case limit exceeds full matrix");
      matrix.resize(o.limit);
    }
    std::cout << "MATRIX collective=" << o.collectiveName << " ranks=" << o.gpus << " algo=" << o.algo
              << " proto=" << o.proto << " planned=" << matrix.size() << " full=" << (o.limit == 0) << '\n'
              << std::flush;
    Devices devices(o.gpus);
    devices.initialize();
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < matrix.size(); ++i)
      runCase(devices, o, matrix[i], static_cast<int>(i), o.injectError && i + 1 == matrix.size());
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    devices.close();
    std::cout << "PASS collective=" << o.collectiveName << " ranks=" << o.gpus << " planned=" << matrix.size()
              << " completed=" << matrix.size() << " body_seconds=" << seconds << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
