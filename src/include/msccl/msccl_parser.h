/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCL_PARSER_H_
#define MSCCL_PARSER_H_

#include "nccl.h"
#include "debug.h"
#include "checks.h"
#include <stdlib.h>

#include "msccl/msccl_struct.h"

// A few constraints to make the implementation easy
#define MAX_STR_LEN 255
#define MAX_ATTR_COUNT 16
#define MAX_SUBS 1024
#define MAX_NODES 4096

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct mscclXmlNode {
  char name[MAX_STR_LEN+1];
  struct {
    char key[MAX_STR_LEN+1];
    char value[MAX_STR_LEN+1];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
  struct mscclXmlNode* parent;
  struct mscclXmlNode* subs[MAX_SUBS];
  int nSubs;
};

struct mscclXml {
  struct mscclXmlNode nodes[MAX_NODES];
  int maxIndex;
};

static ncclResult_t mscclXmlGetAttrIndex(struct mscclXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a=0; a<nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttr(struct mscclXmlNode* node, const char* attrName, const char** value) {
  int index;
  NCCLCHECK(mscclXmlGetAttrIndex(node, attrName, &index));
  *value = index == -1 ? NULL : node->attrs[index].value;
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttrStr(struct mscclXmlNode* node, const char* attrName, const char** value) {
  NCCLCHECK(mscclXmlGetAttr(node, attrName, value));
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t mscclXmlGetAttrInt(struct mscclXmlNode* node, const char* attrName, int* value) {
  const char* str;
  NCCLCHECK(mscclXmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t mscclXmlGetAttrInt64(struct mscclXmlNode* node, const char* attrName, int64_t* value) {
  const char* str;
  NCCLCHECK(mscclXmlGetAttrStr(node, attrName, &str));
  *value = strtoll(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t mscclXmlFindTag(struct mscclXml* xml, const char* tagName, struct mscclXmlNode** node) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct mscclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoFromXmlFile(const char* xmlGraphFile, struct mscclAlgo* algo, int rank);

ncclResult_t mscclGetAlgoMetaFromXmlFile(const char* xmlGraphFile, struct mscclAlgoMeta* algoMeta);

#endif
