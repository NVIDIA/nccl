/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "debug.h"
#include <unistd.h>
#include <string.h>

ncclResult_t getHostName(char* hostname, int maxlen) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != '.') && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

uint64_t getHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname) $(readlink /proc/self/ns/uts)
 */
uint64_t getHostHash(void) {
  char uname[1024];
  // Start off with the hostname
  (void) getHostName(uname, sizeof(uname));
  int hlen = strlen(uname);
  int len = readlink("/proc/self/ns/uts", uname+hlen, sizeof(uname)-1-hlen);
  if (len < 0) len = 0;

  uname[hlen+len]='\0';
  TRACE(INIT,"unique hostname '%s'", uname);

  return getHash(uname);
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long) getpid());
  int plen = strlen(pname);
  int len = readlink("/proc/self/ns/pid", pname+plen, sizeof(pname)-1-plen);
  if (len < 0) len = 0;

  pname[plen+len]='\0';
  TRACE(INIT,"unique PID '%s'", pname);

  return getHash(pname);
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;
  // Ignore "^" prefix, will be detected outside of this function
  if (ptr[0] == '^') ptr++;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchPrefix(const char* string, const char* prefix) {
  return (strncmp(string, prefix, strlen(prefix)) == 0);
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}


bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchPrefix(string, ifList[i].prefix)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}
