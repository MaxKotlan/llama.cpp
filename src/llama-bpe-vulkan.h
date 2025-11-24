#pragma once

#include "llama.h"

#include <memory>
#include <string>
#include <vector>

struct llama_vocab;

struct llama_bpe_vulkan {
    virtual ~llama_bpe_vulkan() = default;
    virtual bool encode(const std::vector<std::string> & words, std::vector<llama_token> & out_tokens) = 0;
};

// Create a Vulkan-backed BPE tokenizer helper.
// Returns nullptr if Vulkan is unavailable or the vocab cannot be mapped to token ids.
std::unique_ptr<llama_bpe_vulkan> llama_bpe_vulkan_create(const llama_vocab & vocab, int max_word_len);

// Encode a batch of already pre-tokenized words. Returns false on failure and leaves output untouched.
bool llama_bpe_vulkan_encode(llama_bpe_vulkan & engine, const std::vector<std::string> & words, std::vector<llama_token> & out_tokens);
