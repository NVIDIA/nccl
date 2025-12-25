#ifndef INSPECTOR_JSON_H_
#define INSPECTOR_JSON_H_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef enum {
  JSON_NONE, // A pseudo-state meaning that the document is empty
  JSON_KEY,
  JSON_OBJECT_EMPTY,
  JSON_OBJECT_SOME,
  JSON_LIST_EMPTY,
  JSON_LIST_SOME,
} jsonState_t;

typedef enum {
  jsonSuccess,
  jsonFileError,
  jsonUnknownStateError,
  jsonEmptyStateError,
  jsonExpectedNonNoneStateError,
  jsonStringOverflowError,
  jsonStringBadChar,
  jsonMemoryError,
  jsonLockError,
} jsonResult_t;

const char *jsonErrorString(jsonResult_t res);

typedef struct jsonFileOutput jsonFileOutput;

jsonResult_t jsonLockOutput(jsonFileOutput *jfo);

jsonResult_t jsonUnlockOutput(jsonFileOutput *jfo);

jsonResult_t jsonInitFileOutput(jsonFileOutput **jfo,
                                const char *outfile);

jsonResult_t jsonFinalizeFileOutput(jsonFileOutput *jfo);

jsonResult_t jsonNewline(jsonFileOutput *jfo);
jsonResult_t jsonFlushOutput(jsonFileOutput *jfo);

// Emit a key and separator. Santize the key.
// This is only acceptable if the top state is an object
// Emit a ',' separator of we aren't the first item.
jsonResult_t jsonKey(jsonFileOutput *jfo, const char *name);

// Start an object
jsonResult_t jsonStartObject(jsonFileOutput *jfo);

// Close an object
jsonResult_t jsonFinishObject(jsonFileOutput *jfo);

// Start a list
jsonResult_t jsonStartList(jsonFileOutput *jfo);

// Close a list
jsonResult_t jsonFinishList(jsonFileOutput *jfo);

// Emit a null value
jsonResult_t jsonNull(jsonFileOutput *jfo);

// Write a (sanititzed) string
jsonResult_t jsonStr(jsonFileOutput *jfo, const char *str);

// Write a bool as "true" or "false" strings.
jsonResult_t jsonBool(jsonFileOutput *jfo, bool val);

// Write an integer value
jsonResult_t jsonInt(jsonFileOutput *jfo, const int val);

//Write an unsigned int value
jsonResult_t jsonUint32(jsonFileOutput *jfo, const uint32_t val);

// Write an integer value
jsonResult_t jsonUint64(jsonFileOutput *jfo, const uint64_t val);

// Write a size_t value
jsonResult_t jsonSize_t(jsonFileOutput *jfo, const size_t val);

// Write a double value
jsonResult_t jsonDouble(jsonFileOutput *jfo, const double val);

#endif  // INSPECTOR_JSON_H_
