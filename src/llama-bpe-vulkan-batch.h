#pragma once

#include "llama.h"
#include <memory>
#include <vector>
#include <string>

struct llama_vocab;

// Create a Vulkan-backed BPE tokenizer helper.
std::unique_ptr<llama_bpe_vulkan> llama_bpe_vulkan_create(const llama_vocab & vocab, int max_word_len);
bool llama_bpe_vulkan_encode(llama_bpe_vulkan & engine, const std::vector<std::string> & words, std::vector<llama_token> & out_tokens);

// Batched encode: flattens multiple prompts (each split into words) and runs one Vulkan dispatch.
// Returns false if any prompt exceeds max_word_len or a Vulkan failure occurs.
bool llama_bpe_vulkan_encode_batch(
    llama_bpe_vulkan & engine,
    const std::vector<std::vector<std::string>> & batch_words,
    std::vector<std::vector<llama_token>> & out_tokens_batch);
