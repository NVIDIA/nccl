/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include "event.h"

int taskEventQueueEmpty(struct group* g) {
  return g->eventHead == NULL;
}

void taskEventQueueEnqueue(struct group* g, struct taskEventBase* event) {
  event->next = NULL;
  if (g->eventHead) g->eventTail->next = event;
  else g->eventHead = event;
  g->eventTail = event;
}

struct taskEventBase* taskEventQueueHead(struct group* g) {
  return g->eventHead;
}

struct taskEventBase* taskEventQueueDequeue(struct group* g) {
  struct taskEventBase* tmp = g->eventHead;
  g->eventHead = g->eventHead->next;
  if (g->eventHead == NULL) g->eventTail = NULL;
  return tmp;
}
