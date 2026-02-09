/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <netdb.h>
#include <unistd.h>

#include "nccl.h"
#define NCCL_RAS_CLIENT // Only pull client-specific definitions from the header file below.
#include "ras_internal.h"

#define STR2(v) #v
#define STR(v) STR2(v)

// Local timeout increment compared to the '-t' argument, in seconds.
#define TIMEOUT_INCREMENT 1

static const char* hostName = "localhost";
static const char* port = STR(NCCL_RAS_CLIENT_PORT);
static int timeout = -1;
static bool verbose = false;
static bool monitorMode = false;
static const char* format = nullptr;
static const char* events = nullptr;
static int sock = -1;

static void printUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [OPTION]...\n"
          "Query the state of a running NCCL job.\n"
          "\nOptions:\n"
          "  -f, --format=FMT    Output format: text or json (text by default)\n"
          "  -h, --host=HOST     Host name or IP address of the RAS client socket of the\n"
          "                      NCCL job to connect to (localhost by default)\n"
          "  -m, --monitor[=GROUPS] Monitor mode: continuously watch for peer changes.\n"
          "                      Optional GROUPS: lifecycle, trace, all, or\n"
          "                      combinations like lifecycle,trace (lifecycle by default)\n"
          "  -p, --port=PORT     TCP port of the RAS client socket of the NCCL job\n"
          "                      (" STR(NCCL_RAS_CLIENT_PORT) " by default)\n"
          "  -t, --timeout=SECS  Maximum time for the local NCCL process to wait for\n"
          "                      responses from other NCCL processes\n"
          "                      (" STR(RAS_COLLECTIVE_LEG_TIMEOUT_SEC) " secs by default; 0 disables the timeout)\n"
          "  -v, --verbose       Increase the verbosity level of the RAS output\n"
          "      --help          Print this help and exit\n"
          "      --version       Print the version number and exit\n", argv0);
}

static void parseArgs(int argc, char** argv) {
  int c;
  int optIdx = 0;
  struct option longOpts[] = {
    {"format",  required_argument, NULL, 'f'},
    {"help",    no_argument,       NULL, 'e'},
    {"host",    required_argument, NULL, 'h'},
    {"monitor", optional_argument, NULL, 'm'},
    {"port",    required_argument, NULL, 'p'},
    {"timeout", required_argument, NULL, 't'},
    {"verbose", no_argument,       NULL, 'v'},
    {"version", no_argument,       NULL, 'r'},
    {0}
  };

  while ((c = getopt_long(argc, argv, "f:h:m::p:t:v", longOpts, &optIdx)) != -1) {
    switch (c) {
      case 'f':
        format = optarg;
        if (strcasecmp(format, "text") != 0 && strcasecmp(format, "json") != 0) {
          fprintf(stderr, "Invalid format: %s (must be text or json)\n", format);
          exit(1);
        }
        break;
      case 'h':
        hostName = optarg;
        break;
      case 'm':
        monitorMode = true;
        if (optarg) {
          events = optarg;
        }
        break;
      case 'p':
        port = optarg;
        break;
      case 't': {
        char* endPtr = nullptr;
        timeout = strtol(optarg, &endPtr, 10);
        if (timeout < 0 || !endPtr || *endPtr != '\0') {
          fprintf(stderr, "Invalid timeout: %s\n", optarg);
          exit(1);
        }
        break;
      }
      case 'v':
        verbose = true;
        break;
      case 'e':
        printUsage(argv[0]);
        exit(0);
      case 'r':
        fprintf(stderr, "NCCL RAS client version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "."
                STR(NCCL_PATCH) NCCL_SUFFIX "\n");
        exit(0);
      default:
        printUsage(argv[0]);
        exit(1);
    }
  }
}

static ssize_t socketWrite(int fd, const void* buf, size_t count) {
  size_t done = 0;
  do {
    ssize_t ret;
    ret = write(fd, ((const char*)buf)+done, count-done);
    if (ret == -1) {
      if (errno != EINTR)
        return -1;
      continue;
    }
    done += ret;
  } while (done < count);

  return done;
}

// Reads a message from RAS.  Assumes that the message ends with '\n' (will continue reading until the terminating
// newline, unless false is passed as untilNewLine).
// Terminates the buffer with '\0'.  Returns the number of bytes read (excluding the added terminating '\0').
static ssize_t rasRead(int fd, void* buf, size_t count, bool untilNewline = true) {
  char* bufChar = (char*)buf;
  size_t done = 0;
  do {
    ssize_t ret;
    ret = read(fd, bufChar+done, count-1-done);
    if (ret == -1) {
      if (errno != EINTR)
        return -1;
      continue;
    }
    if (ret == 0)
      break; // EOF
    done += ret;
  } while (untilNewline && (done == 0 || bufChar[done-1] != '\n'));
  bufChar[done] = '\0';

  return done;
}

static int connectToNCCL() {
  struct addrinfo hints = {0};
  struct addrinfo* addrInfo = nullptr;
  int ret;
  char msgBuf[1024];
  int bytes;
  struct timeval tv = {TIMEOUT_INCREMENT, 0};

retry:
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if ((ret = getaddrinfo(hostName, port, &hints, &addrInfo)) != 0) {
    fprintf(stderr, "Resolving %s:%s: %s\n", hostName, port, gai_strerror(ret));
    goto fail;
  }
  for (struct addrinfo* ai = addrInfo; ai; ai = ai->ai_next) {
    char hostBuf[NI_MAXHOST], portBuf[NI_MAXSERV];
    int err;
    sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock == -1) {
      perror("socket");
      continue;
    }
    // Initially start with a small, 1-sec timeout to quickly eliminate non-responsive processes...
    if (timeout && (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv) != 0 ||
                    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0)) {
      perror("setsockopt");
      // Non-fatal; fall through.
    }
    if (connect(sock, ai->ai_addr, ai->ai_addrlen) == 0)
      break;
    err = errno;
    if (getnameinfo(ai->ai_addr, ai->ai_addrlen, hostBuf, sizeof(hostBuf), portBuf, sizeof(portBuf),
                    NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
      strcpy(hostBuf, hostName);
      strcpy(portBuf, port);
    }
    fprintf(stderr, "Connecting to %s:%s: %s\n", hostBuf, portBuf, strerror(err));
    close(sock);
    sock = -1;
  }
  freeaddrinfo(addrInfo);
  addrInfo = nullptr;

  if (sock == -1) {
    fprintf(stderr, "Failed to connect to the NCCL RAS service!\n"
            "Please make sure that the NCCL job has the RAS service enabled and that\n"
            "%s.\n",
            (strcmp(hostName, "localhost") || strcmp(port, STR(NCCL_RAS_CLIENT_PORT)) ?
            "the host/port arguments are correct and match NCCL_RAS_ADDR" :
            "the RAS client was started on a node where the NCCL job is running"));
    goto fail;
  }

  // Exchange the RAS client handshake.
  strcpy(msgBuf, "CLIENT PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
  if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      goto timeout;
    }
    perror("write to socket");
    goto fail;
  }
  bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
  if (bytes < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      goto timeout;
    }
    perror("read socket");
    goto fail;
  }
  if (bytes == 0) {
    fprintf(stderr, "NCCL unexpectedly closed the connection\n");
    goto fail;
  }
  if (strncasecmp(msgBuf, "SERVER PROTOCOL ", strlen("SERVER PROTOCOL "))) {
    fprintf(stderr, "Unexpected response from NCCL: %s\n", msgBuf);
    goto fail;
  }
  if (strtol(msgBuf+strlen("SERVER PROTOCOL "), nullptr, 10) != NCCL_RAS_CLIENT_PROTOCOL) {
    fprintf(stderr, "NCCL RAS protocol version mismatch (NCCL: %s; RAS client: %d)!\n"
            "Will try to continue in spite of that...\n", msgBuf+strlen("SERVER PROTOCOL "), NCCL_RAS_CLIENT_PROTOCOL);
  }

  if (timeout >= 0) {
    snprintf(msgBuf, sizeof(msgBuf), "TIMEOUT %d\n", timeout);
    if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        goto timeout;
      }
      perror("write to socket");
      goto fail;
    }
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        goto timeout;
      }
      perror("read socket");
      goto fail;
    }
    if (bytes == 0) {
      fprintf(stderr, "NCCL unexpectedly closed the connection\n");
      goto fail;
    }
    if (strcasecmp(msgBuf, "OK\n")) {
      fprintf(stderr, "Unexpected response from NCCL: %s\n", msgBuf);
      goto fail;
    }
  }
  if (timeout) {
    // Increase the socket timeout to accommodate NCCL timeout.
    tv.tv_sec += (timeout > 0 ? timeout : RAS_COLLECTIVE_LEG_TIMEOUT_SEC) + RAS_COLLECTIVE_EXTRA_TIMEOUT_SEC;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0) {
      perror("setsockopt");
      // Non-fatal; fall through.
    }
  }

  return 0;
fail:
  if (addrInfo)
    freeaddrinfo(addrInfo);
  if (sock != -1)
    (void)close(sock);
  return 1;
timeout:
  fprintf(stderr, "Connection timed out; retrying...\n");
  (void)close(sock);
  goto retry;
}

static int setOutputFormat() {
  char msgBuf[4096];
  int bytes;

  // Only set format if the user explicitly specified it.
  if (format) {
    snprintf(msgBuf, sizeof(msgBuf), "SET FORMAT %s\n", format);
    if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("write to socket");
      return 1;
    }
    // Read response.
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("read socket");
      return 1;
    }
    if (bytes == 0) {
      fprintf(stderr, "NCCL unexpectedly closed the connection\n");
      return 1;
    }
    if (strcasecmp(msgBuf, "OK\n")) {
      fprintf(stderr, "Unexpected response from NCCL: %s\n", msgBuf);
      return 1;
    }
  }
  return 0;
}

static int getNCCLStatus() {
  char msgBuf[4096];
  int bytes;

  // Send the status command.
  snprintf(msgBuf, sizeof(msgBuf), "%sSTATUS\n", (verbose ? "VERBOSE " : ""));
  if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      fprintf(stderr, "Connection timed out\n");
    else
      perror("write to socket");
    return 1;
  }
  for (;;) {
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf), /*untileNewLine*/false);
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("read socket");
      return 1;
    }
    if (bytes == 0) // EOF
      break;
    if (fwrite(msgBuf, 1, bytes, stdout) != bytes) {
      fprintf(stderr, "fwrite to stdout failed!\n");
      return 1;
    }
    if (fflush(stdout) != 0) {
      perror("fflush stdout");
      return 1;
    }
  }
  return 0;
}

static int monitorNCCLEvents() {
  char msgBuf[4096];
  int bytes;
  struct timeval tv = {0, 0}; // No timeout for monitor mode.

  // Send the monitor command with optional event levels.
  if (events) {
    snprintf(msgBuf, sizeof(msgBuf), "MONITOR %s\n", events);
  } else {
    snprintf(msgBuf, sizeof(msgBuf), "MONITOR\n");
  }
  if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      fprintf(stderr, "Connection timed out\n");
    else
      perror("Failed to send monitor command");
    return 1;
  }

  // Wait for initial response confirming monitor mode is activated.
  bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
  if (bytes < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      fprintf(stderr, "Connection timed out\n");
    else
      perror("read socket");
    return 1;
  }
  if (bytes == 0) {
    fprintf(stderr, "Connection closed by server\n");
    return 1;
  }

  if (bytes < 3 || strncasecmp(msgBuf, "OK\n", 3) != 0) {
    fprintf(stderr, "Monitor mode activation failed: %.*s", bytes, msgBuf);
    return 1;
  }

  // Disable receive timeout for monitor mode (wait indefinitely for notifications).
  if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0) {
    perror("Failed to disable socket timeout for monitor mode");
    return 1;
  }

  fprintf(stderr, "RAS Monitor Mode - watching for peer changes (Ctrl+C to exit)...\n");
  fprintf(stderr, "================================================================\n");

  // Find the first newline after "OK" to determine where the response ends.
  char* okEnd = strchr(msgBuf, '\n');
  if (okEnd && okEnd < msgBuf + bytes - 1) {
    // There's data after the OK response, output it.
    int okLen = okEnd - msgBuf + 1;
    int remainingBytes = bytes - okLen;
    if (fwrite(msgBuf + okLen, 1, remainingBytes, stdout) != remainingBytes) {
      fprintf(stderr, "fwrite to stdout failed!\n");
      return 1;
    }
    if (fflush(stdout) != 0) {
      perror("fflush stdout");
      return 1;
    }
  }

  // Continuous monitoring loop.
  for (;;) {
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
    if (bytes < 0) {
      if (errno == EINTR) {
        // Handle Ctrl+C gracefully.
        fprintf(stderr, "\nMonitoring stopped by user.\n");
        break;
      }
      perror("read socket");
      return 1;
    }
    if (bytes == 0) {
      fprintf(stderr, "Connection closed by the NCCL job.\n");
      break;
    }

    if (fwrite(msgBuf, 1, bytes, stdout) != bytes) {
      fprintf(stderr, "fwrite to stdout failed!\n");
      return 1;
    }
    if (fflush(stdout) != 0) {
      perror("fflush stdout");
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  parseArgs(argc, argv);

  if (connectToNCCL())
    return 1;

  // Set the output format.
  if (setOutputFormat() != 0) {
    (void)close(sock);
    return 1;
  }

  int result;
  if (monitorMode) {
    result = monitorNCCLEvents();
  } else {
    result = getNCCLStatus();
  }

  if (result != 0) {
    (void)close(sock);
    return 1;
  }

  if (close(sock) == -1) {
    perror("close socket");
    return 1;
  }
  return 0;
}
