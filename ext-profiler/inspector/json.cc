#include "json.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* jsonErrorString(jsonResult_t res) {
  switch (res) {
  case jsonSuccess:
    return "jsonSuccess";
  case jsonFileError:
    return "jsonFileError";
  case jsonUnknownStateError:
    return "jsonUnknownStateError";
  case jsonEmptyStateError:
    return "jsonEmptyStateError";
  case jsonExpectedNonNoneStateError:
    return "jsonExpectedNonNoneStateError";
  case jsonMemoryError:
    return "jsonMemoryError";
  case jsonStringOverflowError:
    return "jsonStringOverflowError";
  case jsonStringBadChar:
    return "jsonStringBadChar";
  case jsonLockError:
    return "jsonLockError";
  default:
    return "unknown json error";
  }
}

// We use these statics to mantain a stack of states where we are writing.
typedef struct jsonFileOutput {
  jsonState_t* states;
  size_t state_cap; // Allocated stack capacity
  size_t state_n;   // # of items in the stack.
  FILE* fp;
  pthread_mutex_t mutex;
} jsonFileOutput;

jsonResult_t jsonInitFileOutput(jsonFileOutput** jfo, const char* outfile) {
  jsonFileOutput* new_jfo = (jsonFileOutput*)malloc(sizeof(jsonFileOutput));
  if (new_jfo == NULL) {
    return jsonMemoryError;
  }
  if (pthread_mutex_init(&new_jfo->mutex, NULL) != 0) {
    free(new_jfo);
    *jfo = 0;
    return jsonLockError;
  }
  new_jfo->states = NULL;
  new_jfo->state_cap = 0;
  new_jfo->state_n = 0;
  new_jfo->fp = fopen(outfile, "w");
  if (new_jfo->fp == NULL) {
    free(new_jfo);
    *jfo = 0;
    return jsonFileError;
  }
  *jfo = new_jfo;
  return jsonSuccess;
}

jsonResult_t jsonNewline(jsonFileOutput* jfo) {
  fprintf(jfo->fp, "\n");
  return jsonSuccess;
}

jsonResult_t jsonFlushOutput(jsonFileOutput* jfo) {
  fflush(jfo->fp);
  return jsonSuccess;
}

jsonResult_t jsonLockOutput(jsonFileOutput* jfo) {
  if (pthread_mutex_lock(&jfo->mutex) != 0) {
    return jsonLockError;
  }
  return jsonSuccess;
}

jsonResult_t jsonUnlockOutput(jsonFileOutput* jfo) {
  if (pthread_mutex_unlock(&jfo->mutex) != 0) {
    return jsonLockError;
  }
  return jsonSuccess;
}

jsonResult_t jsonFinalizeFileOutput(jsonFileOutput* jfo) {
  // Really should probably complain if we aren't in a valid state

  if (pthread_mutex_destroy(&jfo->mutex) != 0) {
    free(jfo);
    return jsonLockError;
  }
  if (jfo->states != NULL) {
    free(jfo->states);
  }
  jfo->states = NULL;
  jfo->state_cap = 0;
  jfo->state_n = 0;
  if (jfo->fp) {
    fclose(jfo->fp);
    jfo->fp = 0;
  }

  free(jfo);
  return jsonSuccess;
}

static int utf8copy(unsigned char* out, int out_lim, const unsigned char* in) {
  int copy_len;
  if ((in[0] & 0xE0) == 0xC0) {
    // 2-byte sequence
    if ((in[1] & 0xC0) != 0x80 || out_lim < 2) {
      return 0;
    }
    copy_len = 2;
  } else if ((in[0] & 0xF0) == 0xE0) {
    // 3-byte sequence
    if ((in[1] & 0xC0) != 0x80 || (in[2] & 0xC0) != 0x80 || out_lim < 3) {
      return 0;
    }
    copy_len = 3;
  } else if ((in[0] & 0xF8) == 0xF0) {
    // 4-byte sequence
    if ((in[1] & 0xC0) != 0x80 || (in[2] & 0xC0) != 0x80 || (in[3] & 0xC0) != 0x80 || out_lim < 4) {
      return 0;
    }
    copy_len = 4;
  } else {
    // Invalid start byte
    return 0;
  }

  for (int i = 0; i < copy_len; ++i) {
    out[i] = in[i];
  }

  return copy_len;
}

// This tries to sanitize/quote a string from 'in' into 'out',
// assuming 'out' has length 'lim'.  We mainly quote ",/,\,\t,\n, and
// bail if we encounter non-printable stuff or non-ASCII stuff.
// 'in' should be null-terminated, of course.
//
// We return false if we were not able to copy all of 'in', either for
// length reasons or for unhandled characters.
static jsonResult_t sanitizeJson(unsigned char out[], int lim, const unsigned char* in) {
  int c = 0;
  while (*in) {
    if (c + 1 >= lim) {
      out[c] = 0;
      return jsonStringOverflowError;
    }
    switch (*in) {
    case '"':
    case '\\':
    case '/':
    case '\t':
    case '\n':
      if (c + 2 > lim) {
        out[c] = 0;
        return jsonStringOverflowError;
      }

      out[c++] = '\\';
      if (*in == '\n') {
        out[c++] = 'n';
      } else if (*in == '\t') {
        out[c++] = 't';
      } else {
        out[c++] = *in;
      }
      ++in;
      break;
    default:
      if (*in <= 0x1F) {
        out[c] = 0;
        return jsonStringBadChar;
      } else if (*in <= 0x7F) {
        out[c++] = *in;
        ++in;
      } else {
        const int utf8len = utf8copy(out + c, lim - c - 1, in);
        if (utf8len == 0) {
          out[c] = 0;
          return jsonStringBadChar;
        }
        c += utf8len;
        in += utf8len;
      }
      break;
    }
  }
  out[c] = 0;
  return jsonSuccess;
}

static size_t max(size_t a, size_t b) {
  if (a < b) {
    return b;
  }
  return a;
}

// Push state onto the state stack. Reallocate for extra storage if needed.
// Because JSON_NONE is a pseudo-state, don't allow it to be pushed.
static jsonResult_t jsonPushState(jsonFileOutput* jfo, jsonState_t state) {
  if (state == JSON_NONE) {
    return jsonExpectedNonNoneStateError;
  }
  if (jfo->state_cap <= (jfo->state_n + 1)) {
    jfo->state_cap = max((size_t)16, jfo->state_cap * 2);
    jfo->states = (jsonState_t*)realloc(jfo->states, sizeof(jsonState_t) * jfo->state_cap);
    if (jfo->states == 0) {
      return jsonMemoryError;
    }
  }
  jfo->states[jfo->state_n++] = state;
  return jsonSuccess;
}

// Return the current state at the top of the stack
static jsonState_t jsonCurrState(const jsonFileOutput* jfo) {
  if (jfo->state_n == 0) {
    return JSON_NONE;
  }
  return jfo->states[jfo->state_n - 1];
}

// Replace the stack with state (equivalent to a pop & push if stack is not empty)
static jsonResult_t jsonReplaceState(jsonFileOutput* jfo, jsonState_t state) {
  if (state == JSON_NONE) {
    return jsonExpectedNonNoneStateError;
  }
  if (jfo->state_n == 0) {
    return jsonEmptyStateError;
  }
  jfo->states[jfo->state_n - 1] = state;
  return jsonSuccess;
}

// Pop the top state off the stack, or return that the state is empty
static jsonState_t jsonPopState(jsonFileOutput* jfo) {
  if (jfo->state_n == 0) {
    return JSON_NONE;
  }
  return jfo->states[--jfo->state_n];
}

// Emit a key and separator. Santize the key.
// This is only acceptable if the top state is an object
// Emit a ',' separator of we aren't the first item.
jsonResult_t jsonKey(jsonFileOutput* jfo, const char* name) {
  switch (jsonCurrState(jfo)) {
  case JSON_OBJECT_EMPTY:
    jsonReplaceState(jfo, JSON_OBJECT_SOME);
    break;
  case JSON_OBJECT_SOME:
    fprintf(jfo->fp, ",");
    break;
  default:
    return jsonUnknownStateError;
  }
  unsigned char tmp[2048];
  const jsonResult_t res = sanitizeJson(tmp, sizeof(tmp), (const unsigned char*)name);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "\"%s\":", tmp);
  jsonPushState(jfo, JSON_KEY);
  return jsonSuccess;
}

// Helper function for inserting values.
// Only acceptable after keys, top-level, or in lists.
// Emit preceeding ',' if in a list and not first item.
static jsonResult_t jsonValHelper(jsonFileOutput* jfo) {
  switch (jsonCurrState(jfo)) {
  case JSON_LIST_EMPTY:
    jsonReplaceState(jfo, JSON_LIST_SOME);
    break;
  case JSON_LIST_SOME:
    fprintf(jfo->fp, ",");
    break;
  case JSON_KEY:
    jsonPopState(jfo);
    break;
  case JSON_NONE:
    break;
  default:
    return jsonUnknownStateError;
  }
  return jsonSuccess;
}

// Start an object
jsonResult_t jsonStartObject(jsonFileOutput* jfo) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "{");
  return jsonPushState(jfo, JSON_OBJECT_EMPTY);
}

// Close an object
jsonResult_t jsonFinishObject(jsonFileOutput* jfo) {
  switch (jsonPopState(jfo)) {
  case JSON_OBJECT_EMPTY:
  case JSON_OBJECT_SOME:
    break;
  default:
    return jsonUnknownStateError;
  }
  fprintf(jfo->fp, "}");
  return jsonSuccess;
}

// Start a list
jsonResult_t jsonStartList(jsonFileOutput* jfo) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "[");
  return jsonPushState(jfo, JSON_LIST_EMPTY);
}

// Close a list
jsonResult_t jsonFinishList(jsonFileOutput* jfo) {
  switch (jsonPopState(jfo)) {
  case JSON_LIST_EMPTY:
  case JSON_LIST_SOME:
    break;
  default:
    return jsonUnknownStateError;
  }
  fprintf(jfo->fp, "]");
  return jsonSuccess;
}

// Write a null value
jsonResult_t jsonNull(jsonFileOutput* jfo) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "null");
  return jsonSuccess;
}

// Write a (sanititzed) string
jsonResult_t jsonStr(jsonFileOutput* jfo, const char* str) {
  if (str == NULL) {
    jsonNull(jfo);
    return jsonSuccess;
  }
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  unsigned char tmp[2048];
  const jsonResult_t san_res = sanitizeJson(tmp, sizeof(tmp), (const unsigned char*)str);
  if (san_res != jsonSuccess) {
    return san_res;
  }
  fprintf(jfo->fp, "\"%s\"", tmp);
  return jsonSuccess;
}

// Write a bool as "true" or "false" strings.
jsonResult_t jsonBool(jsonFileOutput* jfo, bool val) {
  return jsonStr(jfo, val ? "true" : "false");
}

// Write an integer value
jsonResult_t jsonInt(jsonFileOutput* jfo, const int val) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "%d", val);
  return jsonSuccess;
}

// Write an integer value
jsonResult_t jsonUint32(jsonFileOutput* jfo, const uint32_t val) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "%u", val);
  return jsonSuccess;
}


// Write an integer value
jsonResult_t jsonUint64(jsonFileOutput* jfo, const uint64_t val) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "%lu", val);
  return jsonSuccess;
}

// Write a size_t value
jsonResult_t jsonSize_t(jsonFileOutput* jfo, const size_t val) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  fprintf(jfo->fp, "%zu", val);
  return jsonSuccess;
}

// Write a double value
jsonResult_t jsonDouble(jsonFileOutput* jfo, const double val) {
  const jsonResult_t res = jsonValHelper(jfo);
  if (res != jsonSuccess) {
    return res;
  }
  if (val != val) {
    fprintf(jfo->fp, "\"nan\"");
  } else {
    fprintf(jfo->fp, "%lf", val);
  }
  return jsonSuccess;
}

#ifdef DO_JSON_TEST
// compile with
// gcc json.cc -Iinclude/ -DDO_JSON_TEST -o json_test
// run with:
// ./json_test
// if something fails, it will print out the error
// if it all works, print out "output matches reference"
#define JSONCHECK(expr)                                         \
  do {                                                          \
    const jsonResult_t res = (expr);                            \
    if (res != jsonSuccess) {                                   \
      fprintf(stderr, "jsonError: %s\n", jsonErrorString(res)); \
      exit(1);                                                  \
    }                                                           \
  } while (0)

int main() {

  const char refstr[] =
    "{\"number\":123,\"utfstring\":\"∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ "
    "¬β = ¬(¬α ∨ β),\",\"list\":[\"true\",null,9423812381231,3123111,0.694234]}";

  jsonFileOutput* jfo;
  JSONCHECK(jsonInitFileOutput(&jfo, "test.json"));
  JSONCHECK(jsonStartObject(jfo));
  JSONCHECK(jsonKey(jfo, "number"));
  JSONCHECK(jsonInt(jfo, 123));
  JSONCHECK(jsonKey(jfo, "utfstring"));
  JSONCHECK(
    jsonStr(jfo, "∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β),"));
  JSONCHECK(jsonKey(jfo, "list"));
  JSONCHECK(jsonStartList(jfo));
  JSONCHECK(jsonBool(jfo, true));
  JSONCHECK(jsonNull(jfo));
  JSONCHECK(jsonUint64(jfo, 9423812381231ULL));
  JSONCHECK(jsonSize_t(jfo, 3123111));
  JSONCHECK(jsonDouble(jfo, 0.69423413));
  JSONCHECK(jsonFinishList(jfo));
  JSONCHECK(jsonFinishObject(jfo));
  JSONCHECK(jsonFinalizeFileOutput(jfo));

  FILE* fp = fopen("test.json", "r");

  const size_t reflen = sizeof(refstr) / sizeof(char);

  char buffer[reflen];

  fread(buffer, sizeof(char), reflen, fp);

  fclose(fp);

  if (memcmp(buffer, refstr, reflen) == 0) {
    printf("output matches reference\n");
  } else {
    printf("output    %s\nreference %s\n", buffer, refstr);
    return 1;
  }

  return 0;
}

#endif
