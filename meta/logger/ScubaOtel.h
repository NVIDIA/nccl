// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/String.h>
#include <folly/Optional.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include <glog/logging.h>
#include <memory>

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/sdk/logs/logger_provider.h"
#include "opentelemetry/sdk/logs/logger_provider_factory.h"
#include "opentelemetry/sdk/logs/processor.h"
#include "opentelemetry/logs/provider.h"
#include "opentelemetry/sdk/logs/batch_log_record_processor.h" // @manual
#include "opentelemetry/sdk/logs/simple_log_record_processor_factory.h"
#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_options.h"

/**
 * Otel ScubaData for non-FB infra
 */
class ScubaData {
 public:
  explicit ScubaData(folly::StringPiece dataset);

  size_t addRawData(
      const std::string& dataset,
      const std::string& message,
      folly::Optional<std::chrono::milliseconds> timeout);

  size_t addSample(const std::string& dataset,
      std::unordered_map<std::string, std::string> normalMap,
      std::unordered_map<std::string, int64_t> intMap,
      std::unordered_map<std::string, double> doubleMap);

 private:
  std::string tableName_;
  ::opentelemetry::nostd::shared_ptr<const opentelemetry::context::RuntimeContextStorage> storage_;
  ::opentelemetry::nostd::shared_ptr<::opentelemetry::logs::Logger> logger_;
};

void initLoggerProvider();
