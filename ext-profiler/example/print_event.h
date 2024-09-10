/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PRINT_EVENT_H_
#define PRINT_EVENT_H_

void debugEvent(void* eHandle, const char* tag);
void printEvent(FILE* fh, void* handle);

#endif
