#ifndef COMMSPECS_H_
#define COMMSPECS_H_

#include <string>

#include <folly/Expected.h>
#include <folly/Hash.h>
#include <folly/Unit.h>

/**
 * WARNING!!! This file is used by (CUDA) device code , so some C++ features &
 * folly classes might not be supported.
 */

/* Error type */
enum commResult_t {
  commSuccess = 0,
  commUnhandledCudaError = 1,
  commSystemError = 2,
  commInternalError = 3,
  commInvalidArgument = 4,
  commInvalidUsage = 5,
  commRemoteError = 6,
  commInProgress = 7,
  commNumResults = 8
};

constexpr const char* commResultToString(commResult_t res) {
  switch (res) {
    case commSuccess:
      return "commSuccess";
    case commUnhandledCudaError:
      return "commUnhandledCudaError";
    case commSystemError:
      return "commSystemError";
    case commInternalError:
      return "commInternalError";
    case commInvalidArgument:
      return "commInvalidArgument";
    case commInvalidUsage:
      return "commInvalidUsage";
    case commRemoteError:
      return "commRemoteError";
    case commInProgress:
      return "commInProgress";
    default:
      return "Unknown";
  }
}

template <>
struct fmt::formatter<commResult_t> : fmt::formatter<const char*> {
  // Parse is inherited from formatter<const char*>
  template <typename FormatContext>
  auto format(commResult_t res, FormatContext& ctx) const {
    // Use your commResultToString function to get string representation
    return fmt::formatter<const char*>::format(commResultToString(res), ctx);
  }
};

/* Reduction operation selector */
enum commRedOp_dummy_t { commNumOps_dummy = 5 };
enum commRedOp_t {
  commSum = 0,
  commProd = 1,
  commMax = 2,
  commMin = 3,
  commAvg = 4,
  /* commNumOps: The number of built-in commRedOp_t values. Also
   * serves as the least possible value for dynamic commRedOp_t's
   * as constructed by commRedOpCreate*** functions. */
  commNumOps = 5,
  /* commMaxRedOp: The largest valid value for commRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(commRedOp_t) when compared to previous NCCL versions to
   * maintain ABI compatibility. */
  commMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(commRedOp_dummy_t))
};

constexpr const char* commOpToString(commRedOp_t op) {
  switch (op) {
    case commSum:
      return "commSum";
    case commProd:
      return "commProd";
    case commMax:
      return "commMax";
    case commMin:
      return "commMin";
    case commAvg:
      return "commAvg";
    default:
      return "Unknown";
  }
}

template <>
struct fmt::formatter<commRedOp_t> : fmt::formatter<const char*> {
  template <typename FormatContext>
  auto format(commRedOp_t op, FormatContext& ctx) const {
    // Use your commOpToString function to get string representation
    return fmt::formatter<const char*>::format(commOpToString(op), ctx);
  }
};

/* Comparison operation selector */
enum commCmpOp_t {
  commCmpEQ = 0,
  commCmpGE = 1,
  commCmpLE = 2,
  commNumCmpOps = 3
};

constexpr const char* commOpToString(commCmpOp_t op) {
  switch (op) {
    case commCmpEQ:
      return "commCmpEQ";
    case commCmpGE:
      return "commCmpGE";
    case commCmpLE:
      return "commCmpLE";
    default:
      return "Unknown";
  }
}

template <>
struct fmt::formatter<commCmpOp_t> : fmt::formatter<const char*> {
  template <typename FormatContext>
  auto format(commCmpOp_t op, FormatContext& ctx) const {
    return fmt::formatter<const char*>::format(commOpToString(op), ctx);
  }
};

/* Data types */
enum commDataType_t {
  commInt8 = 0,
  commChar = 0,
  commUint8 = 1,
  commInt32 = 2,
  commInt = 2,
  commUint32 = 3,
  commInt64 = 4,
  commUint64 = 5,
  commFloat16 = 6,
  commHalf = 6,
  commFloat32 = 7,
  commFloat = 7,
  commFloat64 = 8,
  commDouble = 8,
  commBfloat16 = 9,
  commFloat8e4m3 = 10,
  commFloat8e5m2 = 11,
  commNumTypes = 12
};

constexpr const char* commDataTypeToString(commDataType_t dtype) {
  switch (dtype) {
    case commInt8:
      // case commChar:
      return "commChar or commInt8";
    case commUint8:
      return "commUint8";
    case commInt32:
      // case commInt:
      return "commInt32";
    case commUint32:
      return "commUint32";
    case commInt64:
      return "commInt64";
    case commUint64:
      return "commUint64";
    case commFloat16:
      // case commHalf:
      return "commFloat16";
    case commFloat32:
      // case commFloat:
      return "commFloat32";
    case commFloat64:
      // case commDouble:
      return "commFloat64";
    case commBfloat16:
      return "commBfloat16";
    case commFloat8e4m3:
      return "commFloat8e4m3";
    case commFloat8e5m2:
      return "commFloat8e5m2";
    case commNumTypes:
      return "commNumTypes";
    default:
      return "Unknown";
  }
}

template <>
struct fmt::formatter<commDataType_t> : fmt::formatter<const char*> {
  template <typename FormatContext>
  auto format(commDataType_t dtype, FormatContext& ctx) const {
    return fmt::formatter<const char*>::format(
        commDataTypeToString(dtype), ctx);
  }
};

inline int commTypeSize(commDataType_t type) {
  switch (type) {
    case commInt8:
    case commUint8:
    case commFloat8e4m3:
    case commFloat8e5m2:
      return 1;
    case commFloat16:
    case commBfloat16:
      return 2;
    case commInt32:
    case commUint32:
    case commFloat32:
      return 4;
    case commInt64:
    case commUint64:
    case commFloat64:
      return 8;
    default:
      return -1;
  }
}

struct CommLogData {
  uint64_t commId{0};
  uint64_t commHash{0xfaceb00c12345678};
  std::string commDesc{"undefined"};
  int rank{-1};
  int nRanks{-1};

  bool operator==(const CommLogData& other) const {
    return (
        commId == other.commId && commHash == other.commHash &&
        commDesc == other.commDesc && rank == other.rank &&
        nRanks == other.nRanks);
  }

  std::size_t hash() const noexcept;
};

// Not include the other types for legacy reasons. We should add them in the
// namespace in the future.
namespace meta::comms {

enum class CommPattern : uint8_t {
  Ring,
  RingTwice,
  PipelineFrom,
  PipelineTo,
  TreeUp,
  TreeDown,
  TreeUpDown,
  CollnetChain,
  CollnetDirect,
  Nvls,
  NvlsTree,
  PatUp,
  PatDown,
  Send,
  Recv,
  NumPatterns
};

enum class CommFunc : int {
  Broadcast = 0,
  Reduce = 1,
  AllGather = 2,
  ReduceScatter = 3,
  AllReduce = 4,
  SendRecv = 5,
  Send = 6,
  Recv = 7,
  NumFuncs = 8
};

enum class CommAlgo {
  Tree = 0,
  Ring = 1,
  CollNetDirect = 2,
  CollNetChain = 3,
  NVLS = 4,
  NVLSTree = 5,
  PAT = 6,
  NumAlgorithms = 7 // Tree/Ring/CollNet*
};

enum class CommProtocol {
  LL = 0,
  LL128 = 1,
  Simple = 2,
  NumProtocols = 3 // Simple/LL/LL128
};

class CommsError {
 public:
  CommsError(std::string msg, commResult_t code)
      : message(std::move(msg)), errorCode(code) {}

  std::string message;
  commResult_t errorCode;

  // Name for the error class
  static const char* name() {
    return "CommsError";
  }

  bool operator==(const CommsError& other) const;
};

template <typename Value>
using CommsMaybe = folly::Expected<Value, CommsError>;

using CommsMaybeVoid = CommsMaybe<folly::Unit>;

} // namespace meta::comms

#endif // COMMSPECS_H_
