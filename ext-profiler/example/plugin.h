/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PLUGIN_H_
#define PLUGIN_H_

__attribute__((visibility("default"))) int exampleProfilerStart(int eActivationMask, const char* name);
__attribute__((visibility("default"))) int exampleProfilerStop(void);


#endif
