/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#include <limits>
#include <mutex>

#include "../diagnostics.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "diagnostics_checks.h"
#include "diagnostics_checks_common.h"
#include "transport.h"
#include "utils.h"

#define RAS_DIAG_XID_LINE_BYTES 8192
#define RAS_DIAG_XID_KMSG_BYTES (4 * 1024 * 1024)
#define RAS_DIAG_XID_DMESG_BYTES (1024 * 1024)
#define RAS_DIAG_XID_LOG_TAIL_BYTES (4 * 1024 * 1024)
#define RAS_DIAG_XID_CHILD_TIMEOUT_SEC 1
#define RAS_DIAG_XID_MAX_EVENTS 64
#define RAS_DIAG_XID_HOST_NAME_BYTES 64

#define RAS_DIAG_XID_DOC "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
#define RAS_DIAG_XID_ARCHIVE_DOC "https://docs.nvidia.com/deploy/xid-errors/archive/index.html"
#define RAS_DIAG_SXID_DOC "https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html"

typedef enum {
  RAS_DIAG_EVENT_XID = 1,
  RAS_DIAG_EVENT_SXID = 2,
} rasDiagnosticsEventType;

typedef enum {
  RAS_DIAG_SXID_CLASS_UNKNOWN = 0,
  RAS_DIAG_SXID_CLASS_NON_FATAL = 1,
  RAS_DIAG_SXID_CLASS_FATAL = 2,
} rasDiagnosticsSxidClass;

typedef enum {
  RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION = 0,
  RAS_DIAG_XID_GUIDANCE_RESTART_APPLICATION = 1,
  RAS_DIAG_XID_GUIDANCE_RESET_GPU = 2,
  RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION = 3, // All other Xids; no specific guidance.
  RAS_DIAG_XID_GUIDANCE_PRE_AMPERE = 4,
} rasDiagnosticsXidGuidance;

struct rasDiagnosticsXidEvent {
  uint32_t code;
  uint16_t count;
  int16_t link; // -1 for Xids.
  int64_t bdf;
  uint8_t type;
  uint8_t sxidClass;
  bool bdfValid;
  bool countSaturated;
};

struct rasDiagnosticsXidScan {
  struct rasDiagnosticsXidEvent events[RAS_DIAG_XID_MAX_EVENTS];
  uint8_t nStored;
  bool scanned;
  bool truncated;
};

struct rasDiagnosticsXidWindow {
  uint64_t monotonicStartUsec;
  uint64_t monotonicEndUsec;
  time_t realtimeStartSec;
  time_t realtimeEndSec;
};

// Automatic and client-triggered diagnostics therefore share this.
static struct {
  uint64_t monotonicUsec;
  time_t realtimeSec;
  bool initialized;
} rasDiagnosticsXidCursor;

// Each process emits one host scan-status record followed by zero or more event records.
typedef enum {
  RAS_DIAG_XID_RECORD_SCAN_STATUS = 0,
  RAS_DIAG_XID_RECORD_EVENT = 1,
} rasDiagnosticsXidRecordKind;

struct rasDiagnosticsXidScanStatus {
  uint64_t hostHash;
  char hostName[RAS_DIAG_XID_HOST_NAME_BYTES];
  bool scanned;
  bool truncated;
  bool preAmpere;
};

struct rasDiagnosticsXidEventData {
  struct rasDiagnosticsXidEvent value;
  uint64_t hostHash;
};

// Fixed-stride host record gathered through RAS.
struct rasDiagnosticsXidData {
  uint8_t kind;
  union {
    struct rasDiagnosticsXidScanStatus scan;
    struct rasDiagnosticsXidEventData event;
  };
};

static bool rasDiagnosticsXidNow(uint64_t* monotonicUsec, time_t* realtimeSec) {
  struct timespec monotonic;
  struct timespec realtime;

  if (clock_gettime(CLOCK_MONOTONIC, &monotonic) != 0 || clock_gettime(CLOCK_REALTIME, &realtime) != 0) return false;
  *monotonicUsec = (uint64_t)monotonic.tv_sec * 1000000ULL + monotonic.tv_nsec / 1000;
  *realtimeSec = realtime.tv_sec;
  return true;
}

void rasDiagnosticsInit() {
  // Capture the Xid/SXid lower bound before the asynchronous RAS thread starts so that the first scan covers the
  // complete RAS lifetime, including events reported immediately after communicator initialization.
  memset(&rasDiagnosticsXidCursor, 0, sizeof(rasDiagnosticsXidCursor));
  if (!rasDiagnosticsXidNow(&rasDiagnosticsXidCursor.monotonicUsec, &rasDiagnosticsXidCursor.realtimeSec)) {
    WARN("RAS diagnostics could not initialize the Xid/SXid scan interval: %s", strerror(errno));
    return;
  }
  rasDiagnosticsXidCursor.initialized = true;
}

static bool rasDiagnosticsXidCaptureWindow(struct rasDiagnosticsXidWindow* window) {
  uint64_t monotonicUsec;
  time_t realtimeSec;

  if (!rasDiagnosticsXidNow(&monotonicUsec, &realtimeSec)) return false;
  if (!rasDiagnosticsXidCursor.initialized) {
    rasDiagnosticsXidCursor.monotonicUsec = monotonicUsec;
    rasDiagnosticsXidCursor.realtimeSec = realtimeSec;
    rasDiagnosticsXidCursor.initialized = true;
  }

  // Scan the half-open interval [start, end) so consecutive scans do not report an event at their shared boundary
  // twice.
  window->monotonicStartUsec = rasDiagnosticsXidCursor.monotonicUsec;
  window->monotonicEndUsec = monotonicUsec;
  window->realtimeStartSec = rasDiagnosticsXidCursor.realtimeSec;
  window->realtimeEndSec = realtimeSec;
  return true;
}

static bool rasDiagnosticsXidParseBdf(const char* begin, const char* end, int64_t* bdf) {
  char text[64];
  unsigned int domain, bus, device, function = 0;
  int consumed = 0;
  size_t len;

  if (begin == nullptr || end == nullptr || bdf == nullptr || end < begin) return false;
  while (begin < end && (*begin == ' ' || *begin == '\t')) begin++;
  while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) end--;
  if ((size_t)(end - begin) >= 4 && strncasecmp(begin, "PCI:", 4) == 0) begin += 4;
  while (begin < end && (*begin == ' ' || *begin == '\t')) begin++;

  len = (size_t)(end - begin);
  if (len == 0 || len >= sizeof(text)) return false;
  memcpy(text, begin, len);
  text[len] = '\0';

  if (sscanf(text, "%x:%x:%x.%x%n", &domain, &bus, &device, &function, &consumed) != 4 || consumed != (int)len) {
    function = 0;
    consumed = 0;
    if (sscanf(text, "%x:%x:%x%n", &domain, &bus, &device, &consumed) != 3 || consumed != (int)len) return false;
  }
  if (domain > 0xffff || bus > 0xff || device > 0x1f || function > 7) return false;
  *bdf = ((int64_t)domain << 20) | ((int64_t)bus << 12) | ((int64_t)device << 4) | function;
  return true;
}

static bool rasDiagnosticsXidParseCode(const char* afterParen, uint32_t* code, const char** afterCode) {
  char* end = nullptr;
  unsigned long value;

  if (afterParen == nullptr || code == nullptr) return false;
  while (*afterParen == ' ' || *afterParen == '\t' || *afterParen == ':') afterParen++;
  errno = 0;
  value = strtoul(afterParen, &end, 10);
  if (errno != 0 || end == afterParen || value > UINT32_MAX) return false;
  *code = (uint32_t)value;
  if (afterCode != nullptr) *afterCode = end;
  return true;
}

static bool rasDiagnosticsXidParseLine(const char* line, struct rasDiagnosticsXidEvent* event) {
  const char* start;
  const char* close;
  const char* afterCode;

  if (line == nullptr || event == nullptr) return false;
  memset(event, 0, sizeof(*event));
  event->link = -1;

  start = strstr(line, "SXid");
  if (start != nullptr) {
    const char* classification;
    const char* classificationEnd;
    const char* link;
    char* linkEnd = nullptr;
    long linkValue;

    start += strlen("SXid");
    while (*start == ' ' || *start == '\t') start++;
    if (*start != '(') return false;
    start++;
    close = strchr(start, ')');
    if (close == nullptr || !rasDiagnosticsXidParseCode(close + 1, &event->code, &afterCode)) return false;

    // Only the summary line contains both the driver classification and Link. Severity/Data lines are details.
    classification = strchr(afterCode, ',');
    if (classification == nullptr) return false;
    classification++;
    while (*classification == ' ' || *classification == '\t') classification++;
    if (strncasecmp(classification, "Non-fatal", strlen("Non-fatal")) == 0) {
      event->sxidClass = RAS_DIAG_SXID_CLASS_NON_FATAL;
      classificationEnd = classification + strlen("Non-fatal");
    } else if (strncasecmp(classification, "Fatal", strlen("Fatal")) == 0) {
      event->sxidClass = RAS_DIAG_SXID_CLASS_FATAL;
      classificationEnd = classification + strlen("Fatal");
    } else {
      return false;
    }

    while (*classificationEnd == ' ' || *classificationEnd == '\t') classificationEnd++;
    if (*classificationEnd != ',') return false;
    link = classificationEnd + 1;
    while (*link == ' ' || *link == '\t') link++;
    if (strncasecmp(link, "Link", strlen("Link")) != 0 || (link[strlen("Link")] != ' ' && link[strlen("Link")] != '\t'))
      return false;
    link += strlen("Link");
    while (*link == ' ' || *link == '\t') link++;
    errno = 0;
    linkValue = strtol(link, &linkEnd, 10);
    if (errno != 0 || linkEnd == link || linkValue < 0 || linkValue > INT16_MAX) return false;

    event->type = RAS_DIAG_EVENT_SXID;
    event->link = (int16_t)linkValue;
    event->bdfValid = rasDiagnosticsXidParseBdf(start, close, &event->bdf);
    return true;
  }

  start = strstr(line, "Xid");
  if (start == nullptr) return false;
  start += strlen("Xid");
  while (*start == ' ' || *start == '\t') start++;

  event->bdfValid = false;
  if (*start == '(') {
    start++;
    close = strchr(start, ')');
    if (close == nullptr || !rasDiagnosticsXidParseCode(close + 1, &event->code, nullptr)) return false;
    event->bdfValid = rasDiagnosticsXidParseBdf(start, close, &event->bdf);
  } else if (!rasDiagnosticsXidParseCode(start, &event->code, nullptr)) {
    return false;
  }

  event->type = RAS_DIAG_EVENT_XID;
  event->sxidClass = RAS_DIAG_SXID_CLASS_UNKNOWN;
  return true;
}

static bool rasDiagnosticsXidSameEvent(const struct rasDiagnosticsXidEvent* a, const struct rasDiagnosticsXidEvent* b) {
  return a->type == b->type && a->code == b->code && a->sxidClass == b->sxidClass && a->link == b->link &&
         a->bdfValid == b->bdfValid && (!a->bdfValid || a->bdf == b->bdf);
}

static void rasDiagnosticsXidAddEvent(struct rasDiagnosticsXidScan* scan, const struct rasDiagnosticsXidEvent* event) {
  for (int i = 0; i < scan->nStored; i++) {
    if (rasDiagnosticsXidSameEvent(scan->events + i, event)) {
      if (scan->events[i].count == UINT16_MAX) scan->events[i].countSaturated = true;
      else scan->events[i].count++;
      return;
    }
  }
  if (scan->nStored == RAS_DIAG_XID_MAX_EVENTS) {
    scan->truncated = true;
    return;
  }
  scan->events[scan->nStored] = *event;
  scan->events[scan->nStored].count = 1;
  scan->nStored++;
}

static bool rasDiagnosticsXidKmsgTimestamp(const char* line, uint64_t* timestampUsec, const char** message) {
  const char* comma1;
  const char* comma2;
  const char* comma3;
  const char* semicolon;
  char* end = nullptr;
  unsigned long long value;

  comma1 = strchr(line, ',');
  if (comma1 == nullptr) return false;
  comma2 = strchr(comma1 + 1, ',');
  if (comma2 == nullptr) return false;
  comma3 = strchr(comma2 + 1, ',');
  semicolon = strchr(comma2 + 1, ';');
  if (comma3 == nullptr || semicolon == nullptr || comma3 > semicolon) return false;

  errno = 0;
  value = strtoull(comma2 + 1, &end, 10);
  if (errno != 0 || end != comma3) return false;
  *timestampUsec = (uint64_t)value;
  *message = semicolon + 1;
  return true;
}

static void rasDiagnosticsXidScanKmsgLine(const char* line, const struct rasDiagnosticsXidWindow* window,
                                          struct rasDiagnosticsXidScan* scan) {
  struct rasDiagnosticsXidEvent event;
  uint64_t timestampUsec;
  const char* message;

  if (!rasDiagnosticsXidKmsgTimestamp(line, &timestampUsec, &message)) return;
  if (timestampUsec < window->monotonicStartUsec || timestampUsec >= window->monotonicEndUsec) return;
  if (rasDiagnosticsXidParseLine(message, &event)) rasDiagnosticsXidAddEvent(scan, &event);
}

static bool rasDiagnosticsXidDmesgTimestamp(const char* line, uint64_t* timestampUsec, const char** message) {
  const char* begin = strchr(line, '[');
  char* end = nullptr;
  unsigned long long seconds;
  unsigned long long fraction = 0;
  int fractionDigits = 0;

  if (begin == nullptr) return false;
  begin++;
  while (*begin == ' ' || *begin == '\t') begin++;
  errno = 0;
  seconds = strtoull(begin, &end, 10);
  if (errno != 0 || end == begin) return false;
  if (*end == '.') {
    end++;
    while (*end >= '0' && *end <= '9') {
      if (fractionDigits < 6) fraction = fraction * 10 + (unsigned long long)(*end - '0');
      fractionDigits++;
      end++;
    }
  }
  while (fractionDigits < 6) {
    fraction *= 10;
    fractionDigits++;
  }
  if (*end != ']' || seconds > (UINT64_MAX - fraction) / 1000000ULL) return false;
  *timestampUsec = seconds * 1000000ULL + fraction;
  *message = end + 1;
  return true;
}

static void rasDiagnosticsXidScanDmesgText(const char* text, const struct rasDiagnosticsXidWindow* window,
                                           struct rasDiagnosticsXidScan* scan) {
  const char* line = text;
  while (line != nullptr && *line != '\0') {
    char copy[RAS_DIAG_XID_LINE_BYTES];
    const char* end = strchr(line, '\n');
    size_t len = end == nullptr ? strlen(line) : (size_t)(end - line);
    struct rasDiagnosticsXidEvent event;
    uint64_t timestampUsec;
    const char* message;

    if (len >= sizeof(copy)) {
      scan->truncated = true;
    } else {
      memcpy(copy, line, len);
      copy[len] = '\0';
      if (rasDiagnosticsXidDmesgTimestamp(copy, &timestampUsec, &message) &&
          timestampUsec >= window->monotonicStartUsec && timestampUsec < window->monotonicEndUsec &&
          rasDiagnosticsXidParseLine(message, &event))
        rasDiagnosticsXidAddEvent(scan, &event);
    }
    line = end == nullptr ? nullptr : end + 1;
  }
}

static bool rasDiagnosticsXidScanKmsg(const struct rasDiagnosticsXidWindow* window,
                                      struct rasDiagnosticsXidScan* scan) {
  char buffer[16384];
  size_t bytesRead = 0;
  int fd;

  memset(scan, 0, sizeof(*scan));
  fd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd == -1) return false;

  scan->scanned = true;
  (void)lseek(fd, 0, SEEK_SET);
  while (!scan->truncated) {
    ssize_t n = read(fd, buffer, sizeof(buffer) - 1);
    if (n > 0) {
      // /dev/kmsg normally terminates at EAGAIN. This limit only ensures that continuous logging cannot keep
      // the RAS thread in this loop forever; it should not be reached in practice.
      if ((size_t)n > RAS_DIAG_XID_KMSG_BYTES - bytesRead) {
        scan->truncated = true;
        break;
      }
      bytesRead += (size_t)n;
      buffer[n] = '\0';
      rasDiagnosticsXidScanKmsgLine(buffer, window, scan);
      continue;
    }
    if (n == 0 || errno == EAGAIN || errno == EWOULDBLOCK) break;
    if (errno == EINTR) continue;
    if (errno == EPIPE) {
      scan->truncated = true;
      break;
    }
    close(fd);
    return false;
  }
  close(fd);
  return true;
}

static bool rasDiagnosticsXidFormatDmesgTime(time_t value, char* text, size_t textBytes) {
  struct tm local;
  return localtime_r(&value, &local) != nullptr && strftime(text, textBytes, "%Y-%m-%d %H:%M:%S", &local) != 0;
}

static bool rasDiagnosticsXidScanDmesg(const struct rasDiagnosticsXidWindow* window,
                                       struct rasDiagnosticsXidScan* scan) {
  ncclUniquePtr<char> output;
  char command[256];
  char since[32];
  char until[32];
  time_t untilSec = window->realtimeEndSec;
  int status;

  memset(scan, 0, sizeof(*scan));
  // dmesg accepts subsecond bounds, but our fallback wall-clock cursor has whole-second precision. Include the full
  // final second here; the monotonic [start, end) filter below still enforces the exact interval.
  if (untilSec < std::numeric_limits<time_t>::max()) untilSec++;
  if (!rasDiagnosticsXidFormatDmesgTime(window->realtimeStartSec, since, sizeof(since)) ||
      !rasDiagnosticsXidFormatDmesgTime(untilSec, until, sizeof(until)))
    return false;
  int commandBytes = snprintf(command, sizeof(command), "dmesg --color=never --since='%s' --until='%s'", since, until);
  if (commandBytes < 0 || commandBytes >= (int)sizeof(command)) return false;
  if (ncclCalloc(output, RAS_DIAG_XID_DMESG_BYTES) != ncclSuccess) return false;
  status = ncclDiagChildRun(command, RAS_DIAG_XID_CHILD_TIMEOUT_SEC, output.get(), RAS_DIAG_XID_DMESG_BYTES);
  if (status != 0) return false;

  scan->scanned = true;
  scan->truncated = strlen(output.get()) == RAS_DIAG_XID_DMESG_BYTES - 1;
  rasDiagnosticsXidScanDmesgText(output.get(), window, scan);
  return true;
}

static bool rasDiagnosticsXidParseIsoWallTime(const char* line, time_t* eventTime) {
  struct tm tmValue = {};
  const char* end;
  int offsetSeconds = 0;
  bool hasOffset = false;

  end = strptime(line, "%Y-%m-%dT%H:%M:%S", &tmValue);
  if (end == nullptr) end = strptime(line, "%Y-%m-%d %H:%M:%S", &tmValue);
  if (end == nullptr) return false;
  if (*end == '.') {
    end++;
    while (*end >= '0' && *end <= '9') end++;
  }
  size_t suffixBytes = strlen(end);
  if (*end == 'Z') {
    hasOffset = true;
  } else if (*end == '+' || *end == '-') {
    if (suffixBytes < 5 || end[1] < '0' || end[1] > '9' || end[2] < '0' || end[2] > '9') return false;
    int sign = *end == '-' ? -1 : 1;
    int hours = (end[1] - '0') * 10 + (end[2] - '0');
    int minutes;
    if (suffixBytes >= 6 && end[3] == ':' && end[4] >= '0' && end[4] <= '9' && end[5] >= '0' && end[5] <= '9') {
      minutes = (end[4] - '0') * 10 + (end[5] - '0');
    } else if (end[3] >= '0' && end[3] <= '9' && end[4] >= '0' && end[4] <= '9') {
      minutes = (end[3] - '0') * 10 + (end[4] - '0');
    } else {
      return false;
    }
    if (hours > 23 || minutes > 59) return false;
    offsetSeconds = sign * (hours * 3600 + minutes * 60);
    hasOffset = true;
  }
  tmValue.tm_isdst = -1;
  *eventTime = hasOffset ? timegm(&tmValue) - offsetSeconds : mktime(&tmValue);
  return *eventTime != (time_t)-1;
}

static bool rasDiagnosticsXidParseSyslogWallTime(const char* line, time_t now, time_t* eventTime) {
  static const char months[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
  struct tm nowTm;
  struct tm tmValue;
  int month = -1;
  int day, hour, minute, second, consumed = 0;

  if (localtime_r(&now, &nowTm) == nullptr) return false;
  tmValue = nowTm;
  if (strlen(line) < 15) return false;
  for (int i = 0; i < 12; i++) {
    if (strncasecmp(line, months + 3 * i, 3) == 0) {
      month = i;
      break;
    }
  }
  if (month < 0 || sscanf(line + 3, " %d %d:%d:%d%n", &day, &hour, &minute, &second, &consumed) != 4 || consumed <= 0 ||
      day < 1 || day > 31 || hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 60)
    return false;
  tmValue.tm_mon = month;
  tmValue.tm_mday = day;
  tmValue.tm_hour = hour;
  tmValue.tm_min = minute;
  tmValue.tm_sec = second;
  tmValue.tm_isdst = -1;
  *eventTime = mktime(&tmValue);
  if (*eventTime == (time_t)-1) return false;
  // Handle a Dec/Jan year boundary when the log line has no year.
  if (*eventTime > now + 24 * 60 * 60) {
    tmValue.tm_year--;
    *eventTime = mktime(&tmValue);
  }
  return *eventTime != (time_t)-1;
}

static char* rasDiagnosticsXidFgetsUntil(FILE* file, long end, char* line, size_t lineBytes) {
  long position = ftell(file);
  if (position < 0 || position >= end || lineBytes < 2) return nullptr;
  long remaining = end - position;
  int readBytes = remaining < (long)lineBytes - 1 ? (int)remaining + 1 : (int)lineBytes;
  return fgets(line, readBytes, file);
}

static bool rasDiagnosticsXidScanLogFile(const char* path, const struct rasDiagnosticsXidWindow* window,
                                         struct rasDiagnosticsXidScan* scan) {
  FILE* file;
  long end;
  char line[RAS_DIAG_XID_LINE_BYTES];

  file = fopen(path, "r");
  if (file == nullptr) return false;
  memset(scan, 0, sizeof(*scan));
  scan->scanned = true;

  if (fseek(file, 0, SEEK_END) != 0 || (end = ftell(file)) < 0) goto fail;
  if (end > RAS_DIAG_XID_LOG_TAIL_BYTES) {
    if (fseek(file, end - RAS_DIAG_XID_LOG_TAIL_BYTES, SEEK_SET) != 0) goto fail;
    scan->truncated = true;
    // Discard the first potentially partial line without allowing an unbounded allocation.
    while (rasDiagnosticsXidFgetsUntil(file, end, line, sizeof(line)) != nullptr) {
      if (strchr(line, '\n') != nullptr) break;
    }
  } else {
    if (fseek(file, 0, SEEK_SET) != 0) goto fail;
  }
  if (ferror(file)) goto fail;

  while (rasDiagnosticsXidFgetsUntil(file, end, line, sizeof(line)) != nullptr) {
    struct rasDiagnosticsXidEvent event;
    time_t eventTime;
    long position = ftell(file);
    if (position < 0) goto fail;
    if (strchr(line, '\n') == nullptr && position < end) {
      scan->truncated = true;
      do {
        if (rasDiagnosticsXidFgetsUntil(file, end, line, sizeof(line)) == nullptr) break;
        position = ftell(file);
        if (position < 0) goto fail;
      } while (strchr(line, '\n') == nullptr && position < end);
      if (ferror(file)) goto fail;
      continue;
    }
    if (!rasDiagnosticsXidParseIsoWallTime(line, &eventTime) &&
        !rasDiagnosticsXidParseSyslogWallTime(line, window->realtimeEndSec, &eventTime))
      continue;
    // Regular-log timestamps have one-second precision, so include the ending second rather than miss an event.
    if (eventTime < window->realtimeStartSec || eventTime > window->realtimeEndSec) continue;
    if (rasDiagnosticsXidParseLine(line, &event)) rasDiagnosticsXidAddEvent(scan, &event);
  }
  if (ferror(file)) goto fail;
  {
    long position = ftell(file);
    if (position < 0) goto fail;
    if (position < end) scan->truncated = true;
  }
  if (fclose(file) != 0) {
    memset(scan, 0, sizeof(*scan));
    return false;
  }
  return true;

fail:
  fclose(file);
  memset(scan, 0, sizeof(*scan));
  return false;
}

static void rasDiagnosticsXidScanLocal(const struct rasDiagnosticsXidWindow* window,
                                       struct rasDiagnosticsXidScan* scan) {
  static const char* logs[] = {"/var/log/kern.log", "/var/log/messages", "/var/log/syslog"};
  if (rasDiagnosticsXidScanKmsg(window, scan)) return;
  if (rasDiagnosticsXidScanDmesg(window, scan)) return;
  for (const char* path : logs) {
    if (rasDiagnosticsXidScanLogFile(path, window, scan)) return;
  }
  memset(scan, 0, sizeof(*scan));
}

template <size_t N>
static bool rasDiagnosticsXidCodeInSet(uint32_t code, const uint16_t (&values)[N]) {
  for (uint16_t value : values) {
    if (code == value) return true;
  }
  return false;
}

static rasDiagnosticsXidGuidance rasDiagnosticsXidClassify(uint32_t code) {
  static const uint16_t noUserAction[] = {63, 106, 107};
  static const uint16_t restartApplication[] = {8,   11,  13,  25,  31,  32,  39,  40,  41,  60,  68,  69,
                                                70,  71,  72,  75,  76,  77,  80,  82,  83,  84,  85,  86,
                                                88,  89,  94,  96,  97,  98,  99,  100, 101, 102, 103, 104,
                                                105, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139};
  static const uint16_t resetGpu[] = {46, 62, 64, 95, 109, 110, 119, 120, 136, 140, 143, 155, 156, 158};

  if (rasDiagnosticsXidCodeInSet(code, noUserAction)) return RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION;
  if (rasDiagnosticsXidCodeInSet(code, restartApplication)) return RAS_DIAG_XID_GUIDANCE_RESTART_APPLICATION;
  if (rasDiagnosticsXidCodeInSet(code, resetGpu)) return RAS_DIAG_XID_GUIDANCE_RESET_GPU;
  return RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION;
}

static rasDiagnosticsXidGuidance rasDiagnosticsSxidClassify(const struct rasDiagnosticsXidEvent* event) {
  static const uint16_t noUserAction[] = {11012, 11021, 11022, 11023, 12021, 12023, 15008, 15011, 19049, 19055,
                                          19057, 19059, 19062, 19065, 19068, 19071, 22013, 24001, 24002, 24003};

  if (event->sxidClass != RAS_DIAG_SXID_CLASS_NON_FATAL) return RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION;
  return rasDiagnosticsXidCodeInSet(event->code, noUserAction) ? RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION :
                                                                 RAS_DIAG_XID_GUIDANCE_CHECK_DOCUMENTATION;
}

ncclResult_t rasDiagnosticsXidCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data) {
  struct rasDiagnosticsXidScan scan;
  struct rasDiagnosticsXidWindow window;
  ncclUniquePtr<char> records;
  const size_t recordStride = sizeof(struct rasDiagnosticsXidData);
  char hostName[RAS_DIAG_XID_HOST_NAME_BYTES];
  uint64_t hostHash;
  int nRecords;
  bool participates = false;
  bool preAmpere = false;

  if (data == nullptr || ctx == nullptr) {
    WARN("RAS diagnostics Xid/SXid collection received invalid arguments");
    return ncclInternalError;
  }
  memset(data, 0, sizeof(*data));

  // A communicator filter controls participation only. Xid/SXid events are reported per host.
  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);
    for (int i = 0; i < nNcclComms; i++) {
      struct ncclComm* comm = ncclComms[i];
      if (comm == nullptr || !COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) continue;
      if (rasDiagnosticsCommMatchesContext(ctx, comm)) {
        participates = true;
        preAmpere |= comm->compCap < 80;
      }
    }
  }
  if (!participates) return ncclSuccess;

  if (!rasDiagnosticsXidCaptureWindow(&window)) {
    WARN("RAS diagnostics could not capture the Xid/SXid scan interval: %s", strerror(errno));
    return ncclSystemError;
  }
  rasDiagnosticsXidScanLocal(&window, &scan);

  nRecords = 1 + scan.nStored;
  NCCLCHECK(ncclCalloc(records, (size_t)nRecords * recordStride));

  hostHash = getHostHash();
  (void)getHostName(hostName, sizeof(hostName), '\0');
  struct rasDiagnosticsXidData* scanData = (struct rasDiagnosticsXidData*)records.get();
  scanData->kind = RAS_DIAG_XID_RECORD_SCAN_STATUS;
  scanData->scan.hostHash = hostHash;
  snprintf(scanData->scan.hostName, sizeof(scanData->scan.hostName), "%s", hostName);
  scanData->scan.scanned = scan.scanned;
  scanData->scan.truncated = scan.truncated;
  scanData->scan.preAmpere = preAmpere;
  for (int eventIdx = 0; eventIdx < scan.nStored; eventIdx++) {
    struct rasDiagnosticsXidData* eventData =
      (struct rasDiagnosticsXidData*)(records.get() + (size_t)(eventIdx + 1) * recordStride);
    eventData->kind = RAS_DIAG_XID_RECORD_EVENT;
    eventData->event.value = scan.events[eventIdx];
    eventData->event.hostHash = hostHash;
  }

  data->records = records.release();
  data->recordsBytes = (int)((size_t)nRecords * recordStride);
  data->recordStride = (int)recordStride;
  data->nRecords = nRecords;
  // Advance the cursor only after the records have been built successfully. Events generated during this scan are
  // newer than the window end and remain part of the next interval.
  rasDiagnosticsXidCursor.monotonicUsec = window.monotonicEndUsec;
  rasDiagnosticsXidCursor.realtimeSec = window.realtimeEndSec;
  return ncclSuccess;
}

static bool rasDiagnosticsXidEventValid(const struct rasDiagnosticsXidEvent* event) {
  if (event->count == 0 || (event->bdfValid && event->bdf < 0)) return false;
  if (event->type == RAS_DIAG_EVENT_XID) {
    return event->sxidClass == RAS_DIAG_SXID_CLASS_UNKNOWN && event->link == -1;
  }
  if (event->type == RAS_DIAG_EVENT_SXID) {
    return (event->sxidClass == RAS_DIAG_SXID_CLASS_NON_FATAL || event->sxidClass == RAS_DIAG_SXID_CLASS_FATAL) &&
           event->link >= 0;
  }
  return false;
}

struct rasDiagnosticsXidInstance {
  struct rasDiagnosticsXidEvent event;
  uint64_t hostHash;
  uint16_t count;
  uint8_t guidance;
  bool countSaturated;
};

struct rasDiagnosticsXidHost {
  uint64_t hostHash;
  char hostName[RAS_DIAG_XID_HOST_NAME_BYTES];
  bool scanned;
  bool truncated;
  bool preAmpere;
};

struct rasDiagnosticsXidFinding {
  uint64_t count;
  uint64_t hostHash;
  uint32_t code;
  int firstLink; // Slice in the summarizer's sorted SXid link array.
  int nLinks;
  uint8_t type;
  uint8_t sxidClass;
  uint8_t guidance;
  bool countSaturated;
};

// Three-way comparison used by qsort comparators.
static int rasDiagnosticsXidCompareUint64(uint64_t a, uint64_t b) {
  return a < b ? -1 : (a > b ? 1 : 0);
}

static int rasDiagnosticsXidInstanceCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsXidInstance* a = (const struct rasDiagnosticsXidInstance*)p1;
  const struct rasDiagnosticsXidInstance* b = (const struct rasDiagnosticsXidInstance*)p2;
  int cmp = rasDiagnosticsXidCompareUint64(a->hostHash, b->hostHash);
  if (cmp != 0) return cmp;
  if (a->event.type != b->event.type) return a->event.type < b->event.type ? -1 : 1;
  if (a->event.code != b->event.code) return a->event.code < b->event.code ? -1 : 1;
  if (a->event.sxidClass != b->event.sxidClass) return a->event.sxidClass < b->event.sxidClass ? -1 : 1;
  if (a->event.link != b->event.link) return a->event.link < b->event.link ? -1 : 1;
  if (a->event.bdfValid != b->event.bdfValid) return a->event.bdfValid < b->event.bdfValid ? -1 : 1;
  return a->event.bdfValid ? rasDiagnosticsXidCompareUint64((uint64_t)a->event.bdf, (uint64_t)b->event.bdf) : 0;
}

static bool rasDiagnosticsXidFindingMatchesInstance(const struct rasDiagnosticsXidFinding* finding,
                                                    const struct rasDiagnosticsXidInstance* instance) {
  return finding->hostHash == instance->hostHash && finding->type == instance->event.type &&
         finding->code == instance->event.code && finding->sxidClass == instance->event.sxidClass &&
         finding->guidance == instance->guidance;
}

static int rasDiagnosticsXidFindHost(const struct rasDiagnosticsXidHost* hosts, int nHosts, uint64_t hostHash) {
  for (int hostIdx = 0; hostIdx < nHosts; hostIdx++) {
    if (hosts[hostIdx].hostHash == hostHash) return hostIdx;
  }
  return -1;
}

static int rasDiagnosticsXidHostCompare(const void* p1, const void* p2) {
  const struct rasDiagnosticsXidHost* a = (const struct rasDiagnosticsXidHost*)p1;
  const struct rasDiagnosticsXidHost* b = (const struct rasDiagnosticsXidHost*)p2;
  return rasDiagnosticsXidCompareUint64(a->hostHash, b->hostHash);
}

static const char* rasDiagnosticsSxidClassName(uint8_t sxidClass) {
  if (sxidClass == RAS_DIAG_SXID_CLASS_FATAL) return "Fatal";
  if (sxidClass == RAS_DIAG_SXID_CLASS_NON_FATAL) return "Non-fatal";
  return "classification unavailable";
}

static void rasDiagnosticsXidFormatLinks(const struct rasDiagnosticsXidFinding* finding, const int16_t* links,
                                         char* text, size_t textBytes) {
  int pos = snprintf(text, textBytes, "{");
  for (int linkIdx = 0; linkIdx < finding->nLinks && pos > 0 && (size_t)pos < textBytes; linkIdx++) {
    pos += snprintf(text + pos, textBytes - pos, "%s%d", linkIdx == 0 ? "" : ",", links[finding->firstLink + linkIdx]);
  }
  if (pos > 0 && (size_t)pos < textBytes) snprintf(text + pos, textBytes - pos, "}");
}

static ncclResult_t rasDiagnosticsXidReportFinding(const struct rasDiagnosticsReporter* reporter,
                                                   const struct rasDiagnosticsXidFinding* finding,
                                                   const int16_t* linkIds, const char* hostName) {
  const char* countPrefix = finding->countSaturated ? "at least " : "";
  const char* occurrence = finding->count == 1 && !finding->countSaturated ? "time" : "times";
  const char* tag = finding->guidance == RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION ? RAS_DIAG_TAG_OK : RAS_DIAG_TAG_INFO;

  if (finding->type == RAS_DIAG_EVENT_SXID) {
    char linkText[512];
    rasDiagnosticsXidFormatLinks(finding, linkIds, linkText, sizeof(linkText));
    if (finding->guidance == RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION) {
      return rasDiagnosticsReport(reporter, tag,
                                  "SXid %u detected %s%" PRIu64 " %s "
                                  "(NVSwitch driver reported %s, "
                                  "link(s) %s) on host %s. No user action or further investigation is required "
                                  "for this SXid.",
                                  finding->code, countPrefix, finding->count, occurrence,
                                  rasDiagnosticsSxidClassName(finding->sxidClass), linkText, hostName);
    }
    return rasDiagnosticsReport(reporter, tag,
                                "SXid %u detected %s%" PRIu64 " %s "
                                "(NVSwitch driver reported %s, "
                                "link(s) %s) on host %s. See the NVIDIA Fabric Manager documentation for the "
                                "recommended "
                                "response: %s",
                                finding->code, countPrefix, finding->count, occurrence,
                                rasDiagnosticsSxidClassName(finding->sxidClass), linkText, hostName, RAS_DIAG_SXID_DOC);
  }

  switch ((rasDiagnosticsXidGuidance)finding->guidance) {
  case RAS_DIAG_XID_GUIDANCE_NO_USER_ACTION:
    return rasDiagnosticsReport(reporter, tag,
                                "Xid %u detected %s%" PRIu64 " %s on host %s. "
                                "No user action or "
                                "further investigation is required for this Xid.",
                                finding->code, countPrefix, finding->count, occurrence, hostName);
  case RAS_DIAG_XID_GUIDANCE_RESTART_APPLICATION:
    return rasDiagnosticsReport(reporter, tag,
                                "Xid %u detected %s%" PRIu64 " %s on host %s. "
                                "Restart the affected "
                                "application. For guidance, see the NVIDIA Xid documentation: %s",
                                finding->code, countPrefix, finding->count, occurrence, hostName, RAS_DIAG_XID_DOC);
  case RAS_DIAG_XID_GUIDANCE_RESET_GPU:
    return rasDiagnosticsReport(reporter, tag,
                                "Xid %u detected %s%" PRIu64 " %s on host %s. "
                                "The NVIDIA Xid catalog recommends resetting the affected GPU. "
                                "Follow the documented GPU-reset capabilities "
                                "and limitations: %s",
                                finding->code, countPrefix, finding->count, occurrence, hostName, RAS_DIAG_XID_DOC);
  case RAS_DIAG_XID_GUIDANCE_PRE_AMPERE:
    return rasDiagnosticsReport(reporter, tag,
                                "Xid %u detected %s%" PRIu64 " %s on host %s. "
                                "A pre-Ampere GPU was detected on this host. For guidance, see the archived NVIDIA "
                                "Xid documentation: %s",
                                finding->code, countPrefix, finding->count, occurrence, hostName,
                                RAS_DIAG_XID_ARCHIVE_DOC);
  default:
    return rasDiagnosticsReport(reporter, tag,
                                "Xid %u detected %s%" PRIu64 " %s on host %s. "
                                "For the recommended "
                                "response, see the NVIDIA Xid documentation: %s",
                                finding->code, countPrefix, finding->count, occurrence, hostName, RAS_DIAG_XID_DOC);
  }
}

ncclResult_t rasDiagnosticsXidSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData) {
  ncclResult_t ret = ncclSuccess;
  struct rasDiagnosticsXidHost* hosts = nullptr;
  struct rasDiagnosticsXidInstance* instances = nullptr;
  struct rasDiagnosticsXidFinding* findings = nullptr;
  int16_t* links = nullptr;
  const size_t recordStride = sizeof(struct rasDiagnosticsXidData);
  int nHosts = 0;
  int nInstances = 0;
  int nFindings = 0;
  int nLinks = 0;
  int nUniqueInstances = 0;
  int nRecords;
  bool complete;

  (void)ctx;
  if (reporter == nullptr || reporter->emit == nullptr) {
    WARN("RAS diagnostics Xid/SXid check received invalid reporter");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr || nData < 0 || nData % (int)recordStride != 0) {
    WARN("RAS diagnostics Xid/SXid check received malformed data size %d", nData);
    return ncclInternalError;
  }

  nRecords = nData / (int)recordStride;
  NCCLCHECK(ncclCalloc(&hosts, nRecords));
  NCCLCHECKGOTO(ncclCalloc(&instances, nRecords), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&findings, nRecords), ret, exit);
  NCCLCHECKGOTO(ncclCalloc(&links, nRecords), ret, exit);

  for (int recordIdx = 0; recordIdx < nRecords; recordIdx++) {
    const struct rasDiagnosticsXidData* xidData =
      (const struct rasDiagnosticsXidData*)(data + (size_t)recordIdx * recordStride);

    if (xidData->kind > RAS_DIAG_XID_RECORD_EVENT) {
      WARN("RAS diagnostics Xid/SXid record is malformed");
      ret = ncclInternalError;
      goto exit;
    }
    if (xidData->kind == RAS_DIAG_XID_RECORD_SCAN_STATUS) {
      if (memchr(xidData->scan.hostName, '\0', sizeof(xidData->scan.hostName)) == nullptr) {
        WARN("RAS diagnostics Xid/SXid record is malformed");
        ret = ncclInternalError;
        goto exit;
      }
      int hostIdx = rasDiagnosticsXidFindHost(hosts, nHosts, xidData->scan.hostHash);
      if (hostIdx < 0) {
        struct rasDiagnosticsXidHost* host = hosts + nHosts++;
        host->hostHash = xidData->scan.hostHash;
        memcpy(host->hostName, xidData->scan.hostName, sizeof(host->hostName));
        host->scanned = xidData->scan.scanned;
        host->truncated = xidData->scan.truncated;
        host->preAmpere = xidData->scan.preAmpere;
      } else {
        struct rasDiagnosticsXidHost* host = hosts + hostIdx;
        host->scanned |= xidData->scan.scanned;
        host->truncated |= xidData->scan.truncated;
        host->preAmpere |= xidData->scan.preAmpere;
        if (strcmp(xidData->scan.hostName, host->hostName) < 0)
          memcpy(host->hostName, xidData->scan.hostName, sizeof(host->hostName));
      }
      continue;
    }
    if (!rasDiagnosticsXidEventValid(&xidData->event.value)) {
      WARN("RAS diagnostics Xid/SXid event record is malformed");
      ret = ncclInternalError;
      goto exit;
    }
    struct rasDiagnosticsXidInstance* instance = instances + nInstances++;
    instance->event = xidData->event.value;
    instance->hostHash = xidData->event.hostHash;
    instance->guidance = xidData->event.value.type == RAS_DIAG_EVENT_XID ?
                           (uint8_t)rasDiagnosticsXidClassify(xidData->event.value.code) :
                           (uint8_t)rasDiagnosticsSxidClassify(&xidData->event.value);
    instance->count = xidData->event.value.count;
    instance->countSaturated = xidData->event.value.countSaturated;
  }

  qsort(hosts, nHosts, sizeof(*hosts), rasDiagnosticsXidHostCompare);
  for (int instanceIdx = 0; instanceIdx < nInstances; instanceIdx++) {
    int hostIdx = rasDiagnosticsXidFindHost(hosts, nHosts, instances[instanceIdx].hostHash);
    if (hostIdx < 0) {
      WARN("RAS diagnostics Xid/SXid event has no host scan-status record");
      ret = ncclInternalError;
      goto exit;
    }
    if (instances[instanceIdx].event.type == RAS_DIAG_EVENT_XID && hosts[hostIdx].preAmpere)
      instances[instanceIdx].guidance = (uint8_t)RAS_DIAG_XID_GUIDANCE_PRE_AMPERE;
  }

  qsort(instances, nInstances, sizeof(*instances), rasDiagnosticsXidInstanceCompare);
  for (int instanceIdx = 0; instanceIdx < nInstances; instanceIdx++) {
    struct rasDiagnosticsXidInstance* instance = instances + instanceIdx;
    if (nUniqueInstances == 0 || rasDiagnosticsXidInstanceCompare(instances + nUniqueInstances - 1, instance) != 0) {
      if (nUniqueInstances != instanceIdx) instances[nUniqueInstances] = *instance;
      nUniqueInstances++;
      continue;
    }
    struct rasDiagnosticsXidInstance* destination = instances + nUniqueInstances - 1;
    if (instance->count > destination->count) destination->count = instance->count;
    destination->countSaturated |= instance->countSaturated;
  }
  nInstances = nUniqueInstances;

  for (int instanceIdx = 0; instanceIdx < nInstances; instanceIdx++) {
    const struct rasDiagnosticsXidInstance* instance = instances + instanceIdx;
    struct rasDiagnosticsXidFinding* finding = nFindings == 0 ? nullptr : findings + nFindings - 1;
    if (finding == nullptr || !rasDiagnosticsXidFindingMatchesInstance(finding, instance)) {
      finding = findings + nFindings++;
      finding->hostHash = instance->hostHash;
      finding->type = instance->event.type;
      finding->code = instance->event.code;
      finding->sxidClass = instance->event.sxidClass;
      finding->guidance = instance->guidance;
      finding->firstLink = nLinks;
    }
    finding->count += instance->count;
    finding->countSaturated |= instance->countSaturated;
    // Instances are sorted by finding key and link, so repeated links are adjacent.
    if (instance->event.link >= 0 &&
        (finding->nLinks == 0 || links[finding->firstLink + finding->nLinks - 1] != instance->event.link)) {
      links[nLinks++] = instance->event.link;
      finding->nLinks++;
    }
  }

  complete = nHosts > 0;
  for (int hostIdx = 0; hostIdx < nHosts; hostIdx++) {
    const struct rasDiagnosticsXidHost* host = hosts + hostIdx;
    if (!host->scanned) {
      complete = false;
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "Xid/SXid: no kernel-log source was available on host %s", host->hostName),
                    ret, exit);
    }
    if (host->truncated) {
      complete = false;
      NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_INFO,
                                         "Xid/SXid: recent-event scan was truncated on host %s", host->hostName),
                    ret, exit);
    }
  }

  for (int findingIdx = 0; findingIdx < nFindings; findingIdx++) {
    const struct rasDiagnosticsXidFinding* finding = findings + findingIdx;
    int hostIdx = rasDiagnosticsXidFindHost(hosts, nHosts, finding->hostHash);
    NCCLCHECKGOTO(rasDiagnosticsXidReportFinding(reporter, finding, links, hosts[hostIdx].hostName), ret, exit);
  }

  if (nFindings == 0 && complete) {
    NCCLCHECKGOTO(rasDiagnosticsReport(reporter, RAS_DIAG_TAG_OK,
                                       "Xid/SXid: no events requiring immediate attention across %d host%s", nHosts,
                                       nHosts == 1 ? "" : "s"),
                  ret, exit);
  }

exit:
  free(links);
  free(findings);
  free(instances);
  free(hosts);
  return ret;
}
