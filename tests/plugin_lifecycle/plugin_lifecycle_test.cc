/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <cassert>
#include <cstdlib>
#include <iostream>

enum ncclResult_t {
  ncclSuccess = 0,
  ncclInternalError = 3,
};

#include "../../src/plugin/plugin_cleanup.h"

enum PluginState {
  PluginStateDisabled = -2,
  PluginStateInitReady = 1,
  PluginStateEnabled = 2,
};

enum NetDeviceType {
  NetDeviceHost = 0,
  NetDeviceGinProxy = 2,
};

struct Comm {
  void* rmaContext = nullptr;
  void* collNetContext = nullptr;
};

struct MockPlugin {
  int initCalls = 0;
  int devicesCalls = 0;
  int getPropertiesCalls = 0;
  int finalizeCalls = 0;
  int devicesResult = 0;
  ncclResult_t getPropertiesResult = ncclSuccess;
  NetDeviceType deviceType = NetDeviceGinProxy;

  ncclResult_t init(void** ctx) {
    initCalls++;
    *ctx = std::malloc(1);
    return *ctx ? ncclSuccess : ncclInternalError;
  }

  ncclResult_t devices(int* ndev) {
    devicesCalls++;
    *ndev = devicesResult;
    return ncclSuccess;
  }

  ncclResult_t getProperties(NetDeviceType* type) {
    getPropertiesCalls++;
    if (getPropertiesResult != ncclSuccess) return getPropertiesResult;
    *type = deviceType;
    return ncclSuccess;
  }

  ncclResult_t finalize(void* ctx) {
    finalizeCalls++;
    std::free(ctx);
    return ncclSuccess;
  }
};

static MockPlugin* activeFinalizePlugin = nullptr;

static ncclResult_t mockFinalize(void* ctx) {
  assert(activeFinalizePlugin != nullptr);
  return activeFinalizePlugin->finalize(ctx);
}

static void rmaPluginInitBeforeFix(Comm* comm, MockPlugin* plugin, PluginState* state) {
  int ndev;
  if (*state >= PluginStateInitReady) {
    if (plugin->init(&comm->rmaContext) != ncclSuccess) *state = PluginStateDisabled;
  }
  if (*state == PluginStateInitReady) {
    if (plugin->devices(&ndev) != ncclSuccess || ndev <= 0) *state = PluginStateDisabled;
    else *state = PluginStateEnabled;
  }
}

static void rmaPluginInitAfterFix(Comm* comm, MockPlugin* plugin, PluginState* state) {
  int ndev;
  bool rmaInitCompleted = false;
  if (*state >= PluginStateInitReady) {
    if (plugin->init(&comm->rmaContext) != ncclSuccess) {
      *state = PluginStateDisabled;
    } else {
      rmaInitCompleted = true;
    }
  }
  if (*state == PluginStateInitReady) {
    if (plugin->devices(&ndev) != ncclSuccess || ndev <= 0) {
      if (rmaInitCompleted) {
        activeFinalizePlugin = plugin;
        ncclPluginFinalizeContext(mockFinalize, &comm->rmaContext);
        activeFinalizePlugin = nullptr;
      }
      *state = PluginStateDisabled;
    } else {
      *state = PluginStateEnabled;
    }
  }
}

static void collNetPluginInitBeforeFix(Comm* comm, MockPlugin* plugin, PluginState* state) {
  int ndev;
  if (*state >= PluginStateInitReady) {
    if (plugin->init(&comm->collNetContext) != ncclSuccess) *state = PluginStateDisabled;
  }
  if (*state == PluginStateInitReady) {
    if (plugin->devices(&ndev) != ncclSuccess || ndev <= 0) *state = PluginStateDisabled;
    else *state = PluginStateEnabled;
  }
}

static void collNetPluginInitAfterFix(Comm* comm, MockPlugin* plugin, PluginState* state) {
  int ndev;
  bool collNetInitCompleted = false;
  if (*state >= PluginStateInitReady) {
    if (plugin->init(&comm->collNetContext) != ncclSuccess) {
      *state = PluginStateDisabled;
    } else {
      collNetInitCompleted = true;
    }
  }
  if (*state == PluginStateInitReady) {
    if (plugin->devices(&ndev) != ncclSuccess || ndev <= 0) {
      if (collNetInitCompleted) {
        activeFinalizePlugin = plugin;
        ncclPluginFinalizeContext(mockFinalize, &comm->collNetContext);
        activeFinalizePlugin = nullptr;
      }
      *state = PluginStateDisabled;
    } else {
      *state = PluginStateEnabled;
    }
  }
}

static ncclResult_t rmaV13InitBeforeFix(void** ctx, MockPlugin* plugin) {
  ncclResult_t ret = plugin->init(ctx);
  if (ret != ncclSuccess) return ret;

  NetDeviceType type;
  ret = plugin->getProperties(&type);
  if (ret != ncclSuccess) return ret;
  if (type != NetDeviceGinProxy) return ncclInternalError;
  return ncclSuccess;
}

static ncclResult_t rmaV13InitAfterFix(void** ctx, MockPlugin* plugin) {
  ncclResult_t ret = ncclSuccess;
  void* tmpCtx = nullptr;
  *ctx = nullptr;

  ret = plugin->init(&tmpCtx);
  if (ret != ncclSuccess) return ret;

  NetDeviceType type;
  ret = plugin->getProperties(&type);
  if (ret != ncclSuccess) goto fail;
  if (type != NetDeviceGinProxy) {
    ret = ncclInternalError;
    goto fail;
  }
  *ctx = tmpCtx;
  return ncclSuccess;

fail:
  if (tmpCtx) {
    activeFinalizePlugin = plugin;
    ncclResult_t finalizeRet = ncclPluginFinalizeContext(mockFinalize, &tmpCtx);
    activeFinalizePlugin = nullptr;
    if (ret == ncclSuccess) ret = finalizeRet;
  }
  return ret;
}

static void testRmaDevicesFailureCleanup() {
  {
    Comm comm;
    MockPlugin plugin;
    PluginState state = PluginStateInitReady;
    rmaPluginInitBeforeFix(&comm, &plugin, &state);
    assert(state == PluginStateDisabled);
    assert(plugin.finalizeCalls == 0);
    assert(comm.rmaContext != nullptr);
    std::free(comm.rmaContext);
  }
  {
    Comm comm;
    MockPlugin plugin;
    PluginState state = PluginStateInitReady;
    rmaPluginInitAfterFix(&comm, &plugin, &state);
    assert(state == PluginStateDisabled);
    assert(plugin.finalizeCalls == 1);
    assert(comm.rmaContext == nullptr);
  }
}

static void testCollNetDevicesFailureCleanup() {
  {
    Comm comm;
    MockPlugin plugin;
    PluginState state = PluginStateInitReady;
    collNetPluginInitBeforeFix(&comm, &plugin, &state);
    assert(state == PluginStateDisabled);
    assert(plugin.finalizeCalls == 0);
    assert(comm.collNetContext != nullptr);
    std::free(comm.collNetContext);
  }
  {
    Comm comm;
    MockPlugin plugin;
    PluginState state = PluginStateInitReady;
    collNetPluginInitAfterFix(&comm, &plugin, &state);
    assert(state == PluginStateDisabled);
    assert(plugin.finalizeCalls == 1);
    assert(comm.collNetContext == nullptr);
  }
}

static void testRmaV13ValidationFailureCleanup() {
  {
    void* ctx = nullptr;
    MockPlugin plugin;
    plugin.deviceType = NetDeviceHost;
    ncclResult_t ret = rmaV13InitBeforeFix(&ctx, &plugin);
    assert(ret == ncclInternalError);
    assert(plugin.finalizeCalls == 0);
    assert(ctx != nullptr);
    std::free(ctx);
  }
  {
    void* ctx = nullptr;
    MockPlugin plugin;
    plugin.deviceType = NetDeviceHost;
    ncclResult_t ret = rmaV13InitAfterFix(&ctx, &plugin);
    assert(ret == ncclInternalError);
    assert(plugin.finalizeCalls == 1);
    assert(ctx == nullptr);
  }
}

static void testRmaV13GetPropertiesFailureCleanup() {
  {
    void* ctx = nullptr;
    MockPlugin plugin;
    plugin.getPropertiesResult = ncclInternalError;
    ncclResult_t ret = rmaV13InitBeforeFix(&ctx, &plugin);
    assert(ret == ncclInternalError);
    assert(plugin.finalizeCalls == 0);
    assert(ctx != nullptr);
    std::free(ctx);
  }
  {
    void* ctx = reinterpret_cast<void*>(0x1);
    MockPlugin plugin;
    plugin.getPropertiesResult = ncclInternalError;
    ncclResult_t ret = rmaV13InitAfterFix(&ctx, &plugin);
    assert(ret == ncclInternalError);
    assert(plugin.finalizeCalls == 1);
    assert(ctx == nullptr);
  }
}

int main() {
  testRmaDevicesFailureCleanup();
  testCollNetDevicesFailureCleanup();
  testRmaV13ValidationFailureCleanup();
  testRmaV13GetPropertiesFailureCleanup();
  std::cout << "plugin lifecycle cleanup tests passed\n";
  return 0;
}
