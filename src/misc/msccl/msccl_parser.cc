/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include "core.h"
#include "collectives.h"
#include "msccl/msccl_parser.h"

ncclResult_t mscclXmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(mscclXmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(mscclXmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(mscclXmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(mscclXmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t mscclXmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(mscclXmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return mscclXmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t mscclXmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetNode(FILE* file, struct mscclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(mscclXmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(mscclXmlSkipComment(file, node->name+3, c));
    return mscclXmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(mscclXmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(mscclXmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(mscclXmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*mscclXmlHandlerFunc_t)(FILE*, struct mscclXml*, struct mscclXmlNode*);

struct mscclXmlHandler {
  const char * name;
  mscclXmlHandlerFunc_t func;
};

ncclResult_t mscclXmlLoadSub(FILE* file, struct mscclXml* xml, struct mscclXmlNode* head, struct mscclXmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == MAX_NODES) {
      WARN("Error : XML parser is limited to 1024 nodes");
      return ncclInternalError;
    }
    struct mscclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct mscclXmlNode));
    NCCLCHECK(mscclXmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(mscclXmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

ncclResult_t mscclAlgoXmlStep(FILE* file, struct mscclXml* xml, struct mscclXmlNode* head) {
  NCCLCHECK(mscclXmlLoadSub(file, xml, head, NULL, 1));
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlThreadBlock(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  struct mscclXmlHandler handlers[] = { { "step", mscclAlgoXmlStep } };
  NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

static int currentRank;

ncclResult_t mscclAlgoXmlGpu(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  int thisrank;
  NCCLCHECK(mscclXmlGetAttrInt(head, "id", &thisrank));
  if (thisrank == currentRank) {
    struct mscclXmlHandler handlers[] = { { "tb", mscclAlgoXmlThreadBlock } };
    NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  } else {
    NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, NULL, 0));
  }
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlAlgo(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  struct mscclXmlHandler handlers[] = { { "gpu", mscclAlgoXmlGpu } };
  NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlLoad(const char* xmlFilePath, struct mscclXml* xml, int rank) {
  currentRank = rank;
  FILE* file = fopen(xmlFilePath, "r");
  if (file == NULL) {
    WARN("Could not open MSCCL XML algorithm file %s : %s", xmlFilePath, strerror(errno));
    return ncclSystemError;
  }
  struct mscclXmlHandler handlers[] = { { "algo", mscclAlgoXmlAlgo } };
  xml->maxIndex = 0;
  NCCLCHECK(mscclXmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

ncclResult_t mscclGetBufferType(const char* str, uint8_t* output) {
  if (strcmp(str, "i") == 0) {
    *output = MSCCL_INPUT_BUFFER;
  } else if (strcmp(str, "o") == 0) {
    *output = MSCCL_OUTPUT_BUFFER;
  } else if (strcmp(str, "s") == 0) {
    *output = MSCCL_SCRATCH_BUFFER;
  } else {
    WARN("type of buffer is not supported: %s", str);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclCheckBufferBounds(int bufferType, int offset, int nInputChunks, int nOutputChunks, int nScratchChunks) {
  if (bufferType == MSCCL_INPUT_BUFFER) {
    if (offset < -1 || offset >= nInputChunks) {
      WARN("Incorrect offset set for input buffer: offset: %d maximum allowed: %d", offset, nInputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_OUTPUT_BUFFER) {
    if (offset < -1 || offset >= nOutputChunks) {
      WARN("Incorrect offset set for output buffer: offset: %d maximum allowed: %d", offset, nOutputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_SCRATCH_BUFFER) {
    if (offset < -1 || offset >= nScratchChunks) {
      WARN("Incorrect offset set for scratch buffer: offset: %d maximum allowed: %d", offset, nScratchChunks);
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

ncclResult_t mscclProtocolStrToId(const char *protocol, int *protocolId) {
  if (strcmp(protocol, "Simple") == 0) {
    *protocolId = NCCL_PROTO_SIMPLE;
  } else if (strcmp(protocol, "LL128") == 0) {
    *protocolId = NCCL_PROTO_LL128;
  } else if (strcmp(protocol, "LL") == 0) {
    *protocolId = NCCL_PROTO_LL;
  } else {
    WARN("MSCCL: protocol %s is not supported.", protocol);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoFromXmlFile(const char* str, struct mscclAlgo* algo, int rank) {
  struct mscclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(mscclAlgoXmlLoad(str, xml, rank));

  // zeroing out all entries.
  memset(algo, 0, sizeof(struct mscclAlgo));
  struct mscclXmlNode* topNode;
  NCCLCHECK(mscclXmlFindTag(xml, "algo", &topNode));

  int nChunksPerLoop;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "nchunksperloop", &nChunksPerLoop));
  algo->nChunksPerLoop  = nChunksPerLoop;

  int nChannels;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "nchannels", &nChannels));
  algo->nChannels = nChannels;

  int nGpus;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "ngpus", &nGpus));
  algo->nRanks = nGpus;

  const char* protocol;
  NCCLCHECK(mscclXmlGetAttrStr(topNode, "proto", &protocol));
  NCCLCHECK(mscclProtocolStrToId(protocol, &algo->protocol));

  algo->sizeMultiplier = 1;
  algo->chunkSteps = MSCCL_CHUNKSTEPS;
  algo->sliceSteps = MSCCL_SLICESTEPS;
  const char* coll;
  NCCLCHECK(mscclXmlGetAttrStr(topNode, "coll", &coll));
  if (strcmp(coll, "reduce") == 0) {
    algo->chunkSteps = REDUCE_CHUNKSTEPS;
    algo->sliceSteps = REDUCE_SLICESTEPS;
    algo->func = mscclFuncReduce;
  } else if (strcmp(coll, "broadcast") == 0) {
    algo->chunkSteps = BROADCAST_CHUNKSTEPS;
    algo->sliceSteps = BROADCAST_SLICESTEPS;
    algo->func = mscclFuncBroadcast;
  } else if (strcmp(coll, "allreduce") == 0) {
    algo->chunkSteps = ALLREDUCE_CHUNKSTEPS;
    algo->sliceSteps = ALLREDUCE_SLICESTEPS;
    algo->func = mscclFuncAllReduce;
  } else if (strcmp(coll, "reducescatter") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->chunkSteps = REDUCESCATTER_CHUNKSTEPS;
    algo->sliceSteps = REDUCESCATTER_SLICESTEPS;
    algo->func = mscclFuncReduceScatter;
  } else if (strcmp(coll, "allgather") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->chunkSteps = ALLGATHER_CHUNKSTEPS;
    algo->sliceSteps = ALLGATHER_SLICESTEPS;
    algo->func = mscclFuncAllGather;
  } else if (strcmp(coll, "send") == 0) {
    algo->func = mscclFuncSend;
  } else if (strcmp(coll, "recv") == 0) {
    algo->func = mscclFuncRecv;
  } else if (strcmp(coll, "gather") == 0) {
    algo->func = mscclFuncGather;
  } else if (strcmp(coll, "scatter") == 0) {
    algo->func = mscclFuncScatter;
  } else if (strcmp(coll, "alltoall") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->func = mscclFuncAllToAll;
  } else if (strcmp(coll, "alltoallv") == 0) {
    algo->func = mscclFuncAllToAllv;
  } else {
    WARN("MSCCL: unsupported collective: %s", coll);
    return ncclInvalidUsage;
  }

  int64_t minBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(topNode, "minBytes", &minBytes));
  algo->minBytes = minBytes;

  int64_t maxBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(topNode, "maxBytes", &maxBytes));
  algo->maxBytes = maxBytes;

  int inplace;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "inplace", &inplace));
  algo->inPlace = (bool)inplace;

  int outofplace;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "outofplace", &outofplace));
  algo->outOfPlace = (bool)outofplace;

  algo->hasReduce = false;

  for (int s=0; s<topNode->nSubs; s++) {
    struct mscclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "gpu") == 0) {
      int blockExists[MSCCL_MAX_NUM_THREAD_BLOCKS];
      memset(blockExists, 0, sizeof(int[MSCCL_MAX_NUM_THREAD_BLOCKS]));
      int id, nScratchChunks, nInputChunks, nOutputChunks;
      NCCLCHECK(mscclXmlGetAttrInt(node, "id", &id));
      if (id == rank) {
        NCCLCHECK(mscclXmlGetAttrInt(node, "i_chunks", &nInputChunks));
        NCCLCHECK(mscclXmlGetAttrInt(node, "o_chunks", &nOutputChunks));
        NCCLCHECK(mscclXmlGetAttrInt(node, "s_chunks", &nScratchChunks));
        if (nScratchChunks < 0) {
          WARN("MSCCL: nScratchChunks must be not negative. nScratchChunks: %d", nScratchChunks);
          return ncclInvalidUsage;
        }
        algo->nScratchChunks = nScratchChunks;
        for (int t=0; t<node->nSubs; t++) {
          struct mscclXmlNode* threadBlockNode = node->subs[t];
          if (strcmp(threadBlockNode->name, "tb") == 0) {
            int bid, recvPeer, sendPeer, channelId;
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "id", &bid));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "recv", &recvPeer));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "send", &sendPeer));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "chan", &channelId));
            if (bid < 0) {
              WARN("MSCCL: bid must be not negative. bid: %d", bid);
              return ncclInvalidUsage;
            }
            if (bid >= MSCCL_MAX_NUM_THREAD_BLOCKS) {
              WARN("MSCCL: too many thread blocks are requested. Max thread blocks: %d", MSCCL_MAX_NUM_THREAD_BLOCKS);
              return ncclInvalidUsage;
            }
            if (blockExists[bid]) {
              WARN("MSCCL: duplicate thread block id %d for MSCCL", bid);
              return ncclInvalidUsage;
            }
            blockExists[bid] = 1;

            if (recvPeer == id || sendPeer == id) {
              WARN("MSCCL: peer (%d,%d) and gpu id (%d) must be different", recvPeer, sendPeer, id);
              return ncclInvalidUsage;
            }
            struct mscclThreadBlock* sTB = &algo->mscclTBs[bid];
            sTB->nSteps = 0;
            if (recvPeer < -1 || sendPeer < -1) {
              WARN("MSCCL: wrong recvPeer (%d) or sendPeer (%d) in thread block %d on gpu %d", recvPeer, sendPeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvPeer == id || sendPeer == id) {
              WARN("MSCCL: recvPeer (%d) or sendPeer (%d) for thread block %d cannot be gpu %d", recvPeer, sendPeer, bid, id);
              return ncclInvalidUsage;
            }

            sTB->recvPeer = recvPeer;
            sTB->sendPeer = sendPeer;
            if (channelId < 0 || channelId > MAXCHANNELS) {
              WARN("MSCCL: threadblock %d on GPU %d has an invalid channel %d", bid, id, channelId);
              return ncclInvalidUsage;
            }
            sTB->channelId = channelId;

            // setting the summary of the msccl algorithm in msccl channels
            mscclChannelInfo* mscclChannel = &algo->mscclChannels[sTB->channelId];

            int numDependencies = 0;
            int oldDependencePointer = 0; // Indicator of where the dependencies started for nop

            int oldReductionDstBuffer = -1; // Indicator of last reduction buffer name; -1 means that last one wasn't a compatible reduction
            int oldReductionDstOffset = -1; // Indicator of last reduction buffer index
            int oldReductionSrcBuffer = -1; //
            int numReductions = 0;

            int numTransfers = 0;
            for (int st=0; st<threadBlockNode->nSubs; st++) {
              struct mscclXmlNode* stepNode = threadBlockNode->subs[st];
              if (strcmp(stepNode->name, "step") == 0) {
                int s, srcOffset, dstOffset, dependBid, dependStep, hasDependence, count;
                const char* srcBuffer, * dstBuffer, * type;
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "s", &s));

                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "srcoff", &srcOffset));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "srcbuf", &srcBuffer));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "dstoff", &dstOffset));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "dstbuf", &dstBuffer));

                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "cnt", &count));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "type", &type));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "depid", &dependBid));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "deps", &dependStep));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "hasdep", &hasDependence));

                if (s >= MSCCL_MAX_NUM_STEPS){
                  WARN("MSCCL: too many steps are requested. Max number of steps: %d, requested: %d", MSCCL_MAX_NUM_STEPS, s+1);
                  return ncclInternalError;
                }
                if (s < 0){
                  WARN("MSCCL: step must be positive: step %d", s);
                  return ncclInternalError;
                }

                int hasSend = 0;
                int hasRecv = 0;
                int checkSrc = 0;
                int checkDst = 0;
                int transferType = -1; // -1 indicate a nop
                if (strcmp(type, "s") == 0) {
                  transferType = MSCCL_SEND;
                  hasSend = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "r") == 0) {
                  transferType = MSCCL_RECV;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rcs") == 0) {
                  transferType = MSCCL_RECV_COPY_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rrs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkSrc = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "rrc") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY;
                  hasRecv = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "rrcs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY_SEND;
                  hasRecv = 1;
                  hasSend = 1;
                  checkSrc = 1;
                  checkDst = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "cpy") == 0) {
                  transferType = MSCCL_LOCAL_COPY;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "re") == 0) {
                  transferType = MSCCL_REDUCE;
                  checkSrc = 1;
                  checkDst = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "nop") == 0) {
                  transferType = -1;
                } else {
                  WARN("MSCCL: type of transfer is not supported: %s", type);
                  return ncclInternalError;
                }

                if (dependBid >= 0) {
                  sTB->dependentBid[numDependencies] = dependBid;
                  sTB->dependentStep[numDependencies] = dependStep;
                  numDependencies++;
                }

                uint8_t srcBufferInt = 0;
                uint8_t dstBufferInt = 0;
                NCCLCHECK(mscclGetBufferType(srcBuffer, &srcBufferInt));
                NCCLCHECK(mscclGetBufferType(dstBuffer, &dstBufferInt));

                int continuationOfReductions = 0;
                // Analyze to see if this is in the same list of reductions for them to be chained
                if (transferType == MSCCL_REDUCE) {
                  if (oldReductionDstBuffer == dstBufferInt && oldReductionDstOffset == dstOffset && oldReductionSrcBuffer == srcBufferInt && dependBid == -1) {
                    numTransfers--; // reuse the same transfer
                    continuationOfReductions = 1;
                  } else {
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                  }
                }

                if (transferType != -1) {
                  struct mscclTransmission* mscclTran = &sTB->transmissions[numTransfers];
                  mscclTran->type = transferType;
                  mscclTran->srcOffset = srcOffset;
                  mscclTran->srcBuffer = srcBufferInt;
                  mscclTran->srcOffset = srcOffset;
                  mscclTran->dstBuffer = dstBufferInt;
                  mscclTran->dstOffset = dstOffset;

                  if (count < 0 || count >= MSCCL_MAX_COUNT){
                    WARN("MSCCL: count (%d) must be positive and less than %d", count, MSCCL_MAX_COUNT);
                    return ncclInternalError;
                  }

                  mscclTran->count = count;

                  if (hasSend) {
                    if (sendPeer < 0) {
                      WARN("MSCCL: there is a send in thread block %d on GPU %d without a sendPeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nSendPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL) {
                      WARN("MSCCL: too many sends per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }

                    struct mscclChannelPeerInfo* sendPeerInfo = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
                    sendPeerInfo->nTransmissionsOfCount[count]++;
                  }
                  if (hasRecv) {
                    if (recvPeer < 0) {
                      WARN("MSCCL: there is a recv in thread block %d on GPU %d without a recvPeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nRecvPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL) {
                      WARN("MSCCL: too many recvs per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }
                    struct mscclChannelPeerInfo* recvPeerInfo = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
                    recvPeerInfo->nTransmissionsOfCount[count]++;
                  }

                  if (checkSrc) NCCLCHECK(mscclCheckBufferBounds(mscclTran->srcBuffer, mscclTran->srcOffset, nInputChunks, nOutputChunks, nScratchChunks));
                  if (checkDst) NCCLCHECK(mscclCheckBufferBounds(mscclTran->dstBuffer, mscclTran->dstOffset, nInputChunks, nOutputChunks, nScratchChunks));

                  if (!continuationOfReductions) {
                    mscclTran->dependencePointer = oldDependencePointer;
                    mscclTran->numDependencies = numDependencies - oldDependencePointer;
                    if (mscclTran->numDependencies > 0 && dependBid < 0) {
                      WARN("MSCCL: when there is a chain of dependencies, the last reduction must be a part of the first immediate instruction. Detected for GPU %d, thread block %d, and step %d. XML will be ignored.", id, bid, s);
                      return ncclInvalidUsage;
                    }
                    oldDependencePointer = numDependencies;
                  }

                  // reduction related pointers
                  if (transferType != MSCCL_REDUCE) {
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                    oldReductionSrcBuffer = -1;
                  } else {
                    if (oldReductionDstBuffer == -1) { // if this is the first reduction
                      mscclTran->reductionPointer = numReductions;
                    }
                    sTB->reductionSrcOffsets[numReductions] = mscclTran->srcOffset;
                    numReductions++;
                    mscclTran->numReductions = numReductions - mscclTran->reductionPointer;

                    if (hasDependence || numReductions == MSCCL_MAX_REDUCE_FUSION) {
                      oldReductionDstBuffer = -1;
                      oldReductionDstOffset = -1;
                    } else {
                      oldReductionDstBuffer = mscclTran->dstBuffer;
                      oldReductionDstOffset = mscclTran->dstOffset;
                      oldReductionSrcBuffer = mscclTran->srcBuffer;
                    }
                  }


                  if (hasDependence != 0 && hasDependence != 1) {
                    WARN("MSCCL: hasDependence needs to be 0 or 1, but it was %d", hasDependence);
                    return ncclInternalError;
                  }
                  mscclTran->hasDependence = hasDependence;

                  numTransfers++;
                  sTB->nSteps = numTransfers;
                }
              }
            }

            // finish up mscclChannel calculation

            for (int c = 0; c < MSCCL_MAX_COUNT; c++) {
              struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
              if (sendPeer->nTransmissionsOfCount[c] > 0) {
                sendPeer->existingCounts[sendPeer->nExistingCounts] = c;
                sendPeer->nExistingCounts++;
              }
              struct mscclChannelPeerInfo* recvPeer = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
              if (recvPeer->nTransmissionsOfCount[c] > 0) {
                recvPeer->existingCounts[recvPeer->nExistingCounts] = c;
                recvPeer->nExistingCounts++;
              }
            }

            if (sTB->sendPeer >= 0) {
              mscclChannel->sendPeerInfo[mscclChannel->nSendPeers].peer = sTB->sendPeer;
              mscclChannel->nSendPeers++;
            }
            if (sTB->recvPeer >= 0) {
              mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers].peer = sTB->recvPeer;
              mscclChannel->nRecvPeers++;
            }
          }
        }
        // make sure that thread blocks are in order. Something like 0, 2, 3 is not allowed.
        if (blockExists[0] == 1) {
          algo->nBlocks = 1;
        }
        for (int i = 1; i < MSCCL_MAX_NUM_THREAD_BLOCKS; i++) {
          if (blockExists[i] == 1 && blockExists[i-1] == 0) {
            WARN("MSCCL: thread block %d is missing", i);
            return ncclInvalidUsage;
          }
          if (blockExists[i] == 1) {
            algo->nBlocks = i+1;
          }
        }

      }
    }
  }
  free(xml);
  return ncclSuccess;
}

ncclResult_t mscclXmlLoadSingleNode(FILE* file, struct mscclXmlNode* node) {
  memset(node, 0, sizeof(struct mscclXmlNode));
  return mscclXmlGetNode(file, node);
}

ncclResult_t mscclAlgoMetaXmlLoad(const char* xmlFilePath, struct mscclXmlNode* node) {
  FILE* file = fopen(xmlFilePath, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open MSCCL XML algorithm file %s : %s", xmlFilePath, strerror(errno));
    return ncclSystemError;
  }
  NCCLCHECK(mscclXmlLoadSingleNode(file, node));
  fclose(file);
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoMetaFromXmlFile(const char* str, struct mscclAlgoMeta* algoMeta) {
  struct mscclXmlNode* node;
  node = (struct mscclXmlNode *)malloc(sizeof(struct mscclXmlNode));
  NCCLCHECK(mscclAlgoMetaXmlLoad(str, node));

  algoMeta->filePath = str;

  int nChunksPerLoop;
  NCCLCHECK(mscclXmlGetAttrInt(node, "nchunksperloop", &nChunksPerLoop));
  algoMeta->nChunksPerLoop  = nChunksPerLoop;

  int nGpus;
  NCCLCHECK(mscclXmlGetAttrInt(node, "ngpus", &nGpus));
  algoMeta->nRanks = nGpus;

  const char* coll;
  NCCLCHECK(mscclXmlGetAttrStr(node, "coll", &coll));
  algoMeta->sizeMultiplier = 1;
  if (strcmp(coll, "reduce") == 0) {
    algoMeta->func = mscclFuncReduce;
  } else if (strcmp(coll, "broadcast") == 0) {
    algoMeta->func = mscclFuncBroadcast;
  } else if (strcmp(coll, "allreduce") == 0) {
    algoMeta->func = mscclFuncAllReduce;
  } else if (strcmp(coll, "reducescatter") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncReduceScatter;
  } else if (strcmp(coll, "allgather") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllGather;
  } else if (strcmp(coll, "send") == 0) {
    algoMeta->func = mscclFuncSend;
  } else if (strcmp(coll, "recv") == 0) {
    algoMeta->func = mscclFuncRecv;
  } else if (strcmp(coll, "gather") == 0) {
    algoMeta->func = mscclFuncGather;
  } else if (strcmp(coll, "scatter") == 0) {
    algoMeta->func = mscclFuncScatter;
  } else if (strcmp(coll, "alltoall") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllToAll;
  } else if (strcmp(coll, "alltoallv") == 0) {
    algoMeta->func = mscclFuncAllToAllv;
  } else {
    return ncclInvalidUsage;
  }

  int64_t minBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(node, "minBytes", &minBytes));
  algoMeta->minBytes = minBytes;

  int64_t maxBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(node, "maxBytes", &maxBytes));
  algoMeta->maxBytes = maxBytes;

  int inplace;
  NCCLCHECK(mscclXmlGetAttrInt(node, "inplace", &inplace));
  algoMeta->inPlace = (bool)inplace;

  int outofplace;
  NCCLCHECK(mscclXmlGetAttrInt(node, "outofplace", &outofplace));
  algoMeta->outOfPlace = (bool)outofplace;

  free(node);
  return ncclSuccess;
}
