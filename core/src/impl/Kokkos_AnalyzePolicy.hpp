/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_IMPL_ANALYZE_POLICY_HPP
#define KOKKOS_IMPL_ANALYZE_POLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>  // IndexType
#include <traits/Kokkos_Traits_fwd.hpp>
#include <traits/Kokkos_PolicyTraitAdaptor.hpp>

#include <traits/Kokkos_ExecutionSpaceTrait.hpp>
#include <traits/Kokkos_GraphKernelTrait.hpp>
#include <traits/Kokkos_IndexTypeTrait.hpp>
#include <traits/Kokkos_IterationPatternTrait.hpp>
#include <traits/Kokkos_LaunchBoundsTrait.hpp>
#include <traits/Kokkos_OccupancyControlTrait.hpp>
#include <traits/Kokkos_ScheduleTrait.hpp>
#include <traits/Kokkos_WorkItemPropertyTrait.hpp>
#include <traits/Kokkos_WorkTagTrait.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="AnalyzePolicyBaseTraits"> {{{1

// Mix in the defaults (base_traits) for the traits that aren't yet handled

//------------------------------------------------------------------------------
// <editor-fold desc="MSVC EBO failure workaround"> {{{2

template <class TraitSpecList>
struct KOKKOS_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION AnalyzeExecPolicyBaseTraits;
template <class... TraitSpecifications>
struct KOKKOS_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION
    AnalyzeExecPolicyBaseTraits<type_list<TraitSpecifications...>>
    : TraitSpecifications::base_traits... {};

// </editor-fold> end AnalyzePolicyBaseTraits }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="AnalyzeExecPolicy specializations"> {{{1

//------------------------------------------------------------------------------
// Note: unspecialized, so that the default pathway is to fall back to using
// the PolicyTraitMatcher. See AnalyzeExecPolicyUseMatcher below
// 就是继承的AnalyzeExcePolicyUseMatcher
// 主模板，处理具有Traits非空的情况
// execution_policy_trait_specifications为一个type_list<ExecutionSpaceTraits ...>类型，
// 该类型包含了预定义的Traits类型
template <class Enable, class... Traits>
struct AnalyzeExecPolicy
    : AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                                  Traits...> {
  using base_t =
      AnalyzeExecPolicyUseMatcher<void, execution_policy_trait_specifications,
                                  Traits...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// Ignore void for backwards compatibility purposes, though hopefully no one is
// using this in application code
// 递归忽略Traits含有的void类型
template <class... Traits>
struct AnalyzeExecPolicy<void, void, Traits...>
    : AnalyzeExecPolicy<void, Traits...> {
  using base_t = AnalyzeExecPolicy<void, Traits...>;
  // 继承基类各个构造函数
  // 奇怪的语法 using [typename] Base::Base
  // 继承其他成员函数 using [typename] Base::memFun
  // using t::mem还有在当前作用域引入mem的作用
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// 特例模板，处理不包含Traits的匹配
// 也是一个递归终止条件，可能Traits匹配了一部分TraitSpecifications
// 直接继承BaseTraits
template <>
struct AnalyzeExecPolicy<void>
    : AnalyzeExecPolicyBaseTraits<execution_policy_trait_specifications> {
  // Ensure default constructibility since a converting constructor causes it to
  // be deleted.
  AnalyzeExecPolicy() = default;

  // Base converting constructor and assignment operator: unless an individual
  // policy analysis deletes a constructor, assume it's convertible
  template <class Other>
  AnalyzeExecPolicy(ExecPolicyTraitsWithDefaults<Other> const&) {}

  template <class Other>
  AnalyzeExecPolicy& operator=(ExecPolicyTraitsWithDefaults<Other> const&) {
    return *this;
  }
};

// </editor-fold> end AnalyzeExecPolicy specializations }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="AnalyzeExecPolicyUseMatcher"> {{{1

// We can avoid having to have policies specialize AnalyzeExecPolicy themselves
// by piggy-backing off of the PolicyTraitMatcher that we need to have for
// things like require() anyway. We mixin the effects of the trait using
// the `mixin_matching_trait` nested alias template in the trait specification

// General PolicyTraitMatcher version

// Matching case
// 递归比较Trait和TraitSpe
// 即每次比较execution_policy_trait_specifications中的一个TraitSepc和Traits中一个Traits
// PolicyTaitMathcer在内部使用具体的is_<concept>比较，如:is_excecution_space
// is_execution_space使用宏定义，其中使用detected_t检测某个T是否含有成员类型execution_space
// 故一个类为执行空间类，只需要包含成员类型execution_space
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<
    std::enable_if_t<PolicyTraitMatcher<TraitSpec, Trait>::value>,
    type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
    : TraitSpec::template mixin_matching_trait<
          Trait, AnalyzeExecPolicy<void, Traits...>> {
  using base_t = typename TraitSpec::template mixin_matching_trait<
      Trait, AnalyzeExecPolicy<void, Traits...>>;
  using base_t::base_t;
};

// Non-matching case
// enable_if_t保证TraitSpec和Trait相同时，由于!::value导致ill-formed，该模板匹配失败忽略
template <class TraitSpec, class... TraitSpecs, class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<
    std::enable_if_t<!PolicyTraitMatcher<TraitSpec, Trait>::value>,
    type_list<TraitSpec, TraitSpecs...>, Trait, Traits...>
    : AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>, Trait,
                                  Traits...> {
  using base_t = AnalyzeExecPolicyUseMatcher<void, type_list<TraitSpecs...>,
                                             Trait, Traits...>;
  using base_t::base_t;
};

// No match found case:
template <class>
struct show_name_of_invalid_execution_policy_trait;
// err如果Trait无法匹配任意一个TraitSpec
template <class Trait, class... Traits>
struct AnalyzeExecPolicyUseMatcher<void, type_list<>, Trait, Traits...> {
  static constexpr auto trigger_error_message =
      show_name_of_invalid_execution_policy_trait<Trait>{};
  static_assert(
      /* always false: */ std::is_void<Trait>::value,
      "Unknown execution policy trait. Search compiler output for "
      "'show_name_of_invalid_execution_policy_trait' to see the type of the "
      "invalid trait.");
};

// All traits matched case:
// 递归终止模板-也继承AnalyzeExecPolicy<void>,所以最终都归结到<void>这个特例模板上
// type_list<>即表示预先定义的 execution_policy_trait_specifications 已经被检查完
template <>
struct AnalyzeExecPolicyUseMatcher<void, type_list<>>
    : AnalyzeExecPolicy<void> {
  using base_t = AnalyzeExecPolicy<void>;
  using base_t::base_t;
};

// </editor-fold> end AnalyzeExecPolicyUseMatcher }}}1
//==============================================================================

//------------------------------------------------------------------------------
// Used for defaults that depend on other analysis results
template <class AnalysisResults>
struct ExecPolicyTraitsWithDefaults : AnalysisResults {
  using base_t = AnalysisResults;
  using base_t::base_t;
  // The old code turned this into an integral type for backwards compatibility,
  // so that's what we're doing here. The original comment was:
  //   nasty hack to make index_type into an integral_type
  //   instead of the wrapped IndexType<T> for backwards compatibility
  using index_type = typename std::conditional_t<
      base_t::index_type_is_defaulted,
      Kokkos::IndexType<typename base_t::execution_space::size_type>,
      typename base_t::index_type>::type;
};

//------------------------------------------------------------------------------

constexpr bool warn_if_deprecated(std::false_type) { return true; }
KOKKOS_DEPRECATED_WITH_COMMENT(
    "Invalid WorkTag template argument in execution policy!!")
constexpr bool warn_if_deprecated(std::true_type) { return true; }
#define KOKKOS_IMPL_STATIC_WARNING(...) \
  static_assert(                        \
      warn_if_deprecated(std::integral_constant<bool, __VA_ARGS__>()), "")

// 实际就是继承的AnalyzeExecuPolicy, 这个Defaults是对特殊情况的包装
// 从总体上来说，就是对给到具体某个Policy的模板类型参数(Traits)进行解析
// 如RangePolicy<Serail, TagA> rangp;
// 这里通过递归模板匹配到了ExecutionSpaceTrait和WorkTagTrait
// 此时通过一系列继承为RangePolicy添加了属性execution_space和work_tag类属性（对应Serial和TagA)
// 对于没有给出的TraitSpecs同样需要添加对应的属性，如index_type等，此时添加的是默认属性类故index_type_is_defaulted = true
// 而execution_space_is_defaulted = false；这是通过在具体匹配到某个TraitSpec时通过mixin_matching_trait实现的，而is_defaulted=true则是最后通过一个默认多重继承实现的继承默认值(根类为默认的BaseTraits, mixin在中间继承的)
template <typename... Traits>
struct PolicyTraits
    : ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>> {
  using base_t =
      ExecPolicyTraitsWithDefaults<AnalyzeExecPolicy<void, Traits...>>;
  using base_t::base_t;
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_3
  KOKKOS_IMPL_STATIC_WARNING(!std::is_empty<typename base_t::work_tag>::value &&
                             !std::is_void<typename base_t::work_tag>::value);
#endif
};

#undef KOKKOS_IMPL_STATIC_WARNING

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_IMPL_ANALYZE_POLICY_HPP
