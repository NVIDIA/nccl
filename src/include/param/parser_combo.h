/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_COMBO_H_INCLUDED
#define PARAM_PARSER_COMBO_H_INCLUDED

#include "param/parser_common.h"
#include "param/parser_enum.h"

#include <memory>
#include <string>

namespace nccl {
namespace param {
namespace parser {

template <typename T, size_t N>
struct comboCtx {
  ncclOptionSet<T, N> options;
  ncclParamParser<T> base;
};

template <typename T, size_t N>
ncclResult_t comboResolve(const void* ctx, const char* input, T& out) {
  auto* c = static_cast<const comboCtx<T, N>*>(ctx);
  if (oneOfResolve<T, N>(&c->options, input, out) == ncclSuccess) return ncclSuccess;
  return c->base.resolve(input, out);
}

template <typename T, size_t N>
bool comboValidate(const void* ctx, const T& val) {
  auto* c = static_cast<const comboCtx<T, N>*>(ctx);
  if (oneOfLookup(c->options, val) != nullptr) return true;
  return c->base.validate(val);
}

template <typename T, size_t N>
std::string comboToString(const void* ctx, const T& val) {
  auto* c = static_cast<const comboCtx<T, N>*>(ctx);
  if (const char* name = oneOfLookup(c->options, val)) return name;
  return c->base.toString(val);
}

} // namespace parser
} // namespace param
} // namespace nccl

// ncclParamCombo: extend a base parser with named options for specific values.
// Options are checked first (case-insensitive name match). If no option matches,
// resolve/validate/toString delegate to the base parser.
template <typename T, size_t N>
ncclParamParser<T> ncclParamCombo(ncclParamParser<T> base, ncclOptionSet<T, N> options) {
  using namespace nccl::param::parser;
  std::string d = base.desc + ", or one of:";
  for (const auto& opt : options) {
    d += "\n        ";
    d += opt.name;
    if (opt.desc != nullptr) {
      d += " - ";
      d += opt.desc;
    }
  }
  auto ctx = std::make_shared<comboCtx<T, N>>(comboCtx<T, N>{std::move(options), std::move(base)});
  return {comboResolve<T, N>, comboValidate<T, N>, comboToString<T, N>, std::move(ctx), std::move(d)};
}

#endif /* PARAM_PARSER_COMBO_H_INCLUDED */
