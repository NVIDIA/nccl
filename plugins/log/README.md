# NCCL Log Plugin

A log plugin receives NCCL's own log records as structured data instead of letting NCCL format them
into a line of text and write it to `NCCL_DEBUG_FILE`. It exists so an application can carry NCCL's
logs into its own logging or telemetry system without having to parse them back apart, and without
patching NCCL.

## Interface

A plugin exports one `ncclLogSink_v1_t` under the symbol `ncclLogPlugin_v1`:

```c
typedef struct {
  int           level;    // ncclDebugLogLevel value
  unsigned long subSys;   // ncclDebugLogSubSys bits
  const char*   file;     // call site, or NULL where the level records none
  const char*   func;
  int           line;
  ncclResult_t  code;     // result code at an error origin, otherwise ncclSuccess
  const char*   format;   // format string before arguments are applied
  const char*   message;  // format with arguments applied, no NCCL prefix
  const char*   hostname; // attribution NCCL would otherwise print as a line prefix
  int           pid;
  int           tid;
  int           cudaDev;
} ncclDebugLogRecord_v1_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context);
  ncclResult_t (*onRecord)(void* context, const ncclDebugLogRecord_v1_t* record);
  ncclResult_t (*finalize)(void* context);
} ncclLogSink_v1_t;
```

`init` and `finalize` may be `NULL`. `onRecord` is required.

The same struct is accepted by `ncclSetDebugLogSink()` for an application that links libnccl and can
register a sink directly, so a sink can move between the two without being rewritten.

## Loading

| Variable | Meaning |
|---|---|
| `NCCL_LOG_PLUGIN=<name>` | Loads `libnccl-log-<name>.so` |
| `NCCL_LOG_PLUGIN=<path>` | Loads that library |
| `NCCL_LOG_PLUGIN=none` | Disables |
| unset | No plugin is loaded and nothing is probed |

A plugin is loaded from `ncclInitEnv()`, after the env plugin. Failing to load is not fatal: NCCL keeps
its default output.

There is one sink slot and two possible claimants -- this plugin, and an application calling
`ncclSetDebugLogSink()` directly. The first to claim it wins: if an application registered a sink before
the first NCCL call, the plugin declines to load and says so, rather than displacing it.

A plugin sink is not torn down. NCCL does not finalize it at process exit, because a sink sits on a path
every other thread uses and tearing it down while they may still be logging is worse than letting the
process exit with it installed. `finalize` runs only if the sink is explicitly removed.

## Contract

- `onRecord` may be called concurrently from several threads, and must do its own serialization. It is
  called without NCCL's logging mutex held, so a slow sink does not stall *other logging threads* -- but
  it does hold a shared lock, so it delays `ncclSetDebugLogSink()`, which waits for in-flight dispatches.
  `onRecord` must therefore not block: a sink that can wedge it can wedge registration, and hence
  shutdown.
- `onRecord` must not call back into NCCL. NCCL's logger is the caller.
- Strings in the record are owned by NCCL and valid only for the duration of the call. Copy anything
  you retain.
- Records are delivered only if they pass `NCCL_DEBUG` and `NCCL_DEBUG_SUBSYS`. Set `NCCL_DEBUG=TRACE`
  and `NCCL_DEBUG_SUBSYS=ALL` to receive everything and filter in the sink.
- Records emitted while the plugin library is being opened precede its installation and go to NCCL's
  default output.
- `NCCL_DEBUG_FILE` is still honored and still created. It receives nothing while a sink is installed,
  and receives NCCL's output again if the sink is removed.

## Severity and error codes

Two fields distinguish where an error came from:

| Record | `level` | `code` |
|---|---|---|
| Root cause -- the site that detected the failure | `NCCL_LOG_ERROR` | the `ncclResult_t` about to be returned |
| Re-report of an error raised further down | `NCCL_LOG_WARN` | `ncclSuccess` |
| Noteworthy, not an error | `NCCL_LOG_ATTN` | `ncclSuccess` |

`NCCL_DEBUG=WARN` includes `ERROR`, so raising a message's severity never removes it from a `WARN`
baseline. `NCCL_DEBUG=ERROR` narrows output to root causes alone. Conversion of call sites is
incremental, so that selects the root causes NCCL can currently identify as such, not every error.

This is the distinction that makes a sink useful for failure analysis: capture a stack and emit a
telemetry record when `code != ncclSuccess`, once, at the origin -- rather than once per layer that
re-reports the error on its way out.

## Example

`example/` contains a plugin that prints each record in a `key=value` form.

```bash
cd example && make

# Root causes and warnings, showing which is which
NCCL_LOG_PLUGIN=$PWD/libnccl-log-example.so NCCL_DEBUG=WARN ./your_app

# Root causes only
NCCL_LOG_PLUGIN=$PWD/libnccl-log-example.so NCCL_DEBUG=ERROR ./your_app
```

A failure that is detected in one place and reported again as it unwinds produces output of this
shape -- one origin carrying a code, and the re-reports that follow it:

```
LOG/Plugin: level=ERROR code=1(unhandled cuda error) subsys=0xffffffffffffffff rank=host01:4242:4250 dev=3 \
            at=init.cc:502(commAlloc) event="Cuda failure '%s'" msg="Cuda failure 'out of memory'"
LOG/Plugin: level=WARN subsys=0xffffffffffffffff rank=host01:4242:4250 msg="ncclAsyncJobComplete: job 0x... failed, job error 1"
```

Before this interface existed both lines were `NCCL WARN` with no code attached, and telling the cause
from the symptom meant matching on message text.
