#include "ScubaOtel.h"
#include <folly/json.h>
#include <folly/dynamic.h>
#include <iostream>

const std::string schema_url{"https://opentelemetry.io/schemas/1.2.0"};

namespace otlp      = ::opentelemetry::exporter::otlp;
namespace logs_sdk  = ::opentelemetry::sdk::logs;
namespace logs      = ::opentelemetry::logs;
namespace resource_sdk = ::opentelemetry::sdk::resource;
namespace nostd     = opentelemetry::v1::nostd;

static std::unordered_map<std::string, ScubaData> tableNameToScubaData;

namespace {

nostd::shared_ptr<opentelemetry::context::RuntimeContextStorage> gStorage;

std::string getOtelEndpointFromEnv() {
  const char* endpoint = std::getenv("OTEL_EXPORTER_OTLP_ENDPOINT");
  if (endpoint == nullptr) {
    return "http://localhost:4318/v1";
  }
  std::string str(endpoint);
  if (str.ends_with("v1")) {
    return str;
  }
  return fmt::format("{}/v1", str);
}
}

void initLoggerProvider() {
  otlp::OtlpHttpLogRecordExporterOptions expOptions;
  expOptions.url = fmt::format("{}/logs", getOtelEndpointFromEnv());
  expOptions.content_type = otlp::HttpRequestContentType::kBinary;
  expOptions.ssl_insecure_skip_verify = true;
  auto exporter = otlp::OtlpHttpLogRecordExporterFactory::Create(expOptions);
  auto processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(exporter));

  std::shared_ptr<logs_sdk::LoggerProvider> loggerProvider(
      logs_sdk::LoggerProviderFactory::Create(std::move(processor)));

  const std::shared_ptr<logs::LoggerProvider> &apiProvider = loggerProvider;
  logs::Provider::SetLoggerProvider(apiProvider);
  gStorage = nostd::shared_ptr<opentelemetry::context::ThreadLocalContextStorage>(new opentelemetry::context::ThreadLocalContextStorage());
  opentelemetry::context::RuntimeContext::SetRuntimeContextStorage(gStorage);
}

ScubaData::ScubaData(folly::StringPiece dataset) {
  // Exporter sends the logs out to the backend.
  auto loggerProvider = logs::Provider::GetLoggerProvider();
  tableName_ = fmt::format("fair_{}", dataset);
  logger_ = loggerProvider->GetLogger(
      "ncclx",
      "ncclx",
      "1.0.0",
      schema_url,
      {{"fb.scuba.table", tableName_}});
  // Whold a reference to the global reference to avoid it being destroyed.
  storage_ = opentelemetry::context::RuntimeContext::GetConstRuntimeContextStorage();
}

size_t ScubaData::addSample(const std::string& dataset,
      std::unordered_map<std::string, std::string> normalMap,
      std::unordered_map<std::string, int64_t> intMap,
      std::unordered_map<std::string, double> doubleMap) {
  auto sample = logger_->CreateLogRecord();
  for (const auto& [key, value] : normalMap) {
    sample->SetAttribute(key, value);
  }
  for (const auto& [key, value] : intMap) {
    sample->SetAttribute(key, value);
  }
  for (const auto& [key, value] : doubleMap) {
    sample->SetAttribute(key, value);
  }
  logger_->EmitLogRecord(std::move(sample));
  return 1;
}

std::vector<nostd::string_view> dynamicToVector(const folly::dynamic& dyn) {
  std::vector<nostd::string_view> vec;
  if (dyn.isArray()) {
    vec.reserve(dyn.size());
    for (const auto& elem : dyn) {
      vec.emplace_back(elem.getString());
    }
  }
  return vec;
}

size_t ScubaData::addRawData(
      const std::string& dataset,
      const std::string& message,
      folly::Optional<std::chrono::milliseconds> timeout) {
  auto sample = logger_->CreateLogRecord();
  try {
    folly::dynamic dyn = folly::parseJson(message);
    for (auto& pair: dyn["normal"].items()) {
      sample->SetAttribute(pair.first.asString(), pair.second.asString());
    }
    for (auto& pair: dyn["int"].items()) {
      sample->SetAttribute(pair.first.asString(), pair.second.asInt());
    }
    for (auto& pair: dyn["double"].items()) {
      sample->SetAttribute(pair.first.asString(), pair.second.asDouble());
    }
    std::unordered_map<std::string, std::vector<nostd::string_view>> normVecMap;
    for (auto& pair: dyn["normvector"].items()) {
      normVecMap.insert_or_assign(pair.first.asString(), dynamicToVector(pair.second));
    }
    for (const auto& [key, value] : normVecMap) {
      sample->SetAttribute(key, value);
    }
    std::unordered_map<std::string, std::vector<nostd::string_view>> tagVecMap;
    for (auto& pair: dyn["tags"].items()) {
      tagVecMap.insert_or_assign(pair.first.asString(), dynamicToVector(pair.second));
    }
    for (const auto& [key, value] : tagVecMap) {
      sample->SetAttribute(key, value);
    }
    logger_->EmitLogRecord(std::move(sample));
  } catch (const std::exception& e) {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
  }
  return 1;
}
