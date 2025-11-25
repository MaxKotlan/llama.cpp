#include "llama-bpe-vulkan.h"

#include "llama-vocab.h"
#include "llama-impl.h"

#ifdef LLAMA_BPE_VULKAN

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

struct merge_entry {
    uint32_t left  = 0;
    uint32_t right = 0;
    uint32_t merged = 0;
    uint32_t rank = 0;
};

struct word_meta {
    uint32_t offset = 0;
    uint32_t len = 0;
};

struct link_pair {
    int32_t prev = -1;
    int32_t next = -1;
};

constexpr uint32_t INVALID_SYMBOL = 0xffffffffu;
constexpr float   GROWTH_FACTOR   = 1.5f;

bool read_file(const char * path, std::vector<uint32_t> & data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0 || (size % sizeof(uint32_t)) != 0) {
        return false;
    }

    data.resize(static_cast<size_t>(size) / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(data.data()), size);
    return file.good();
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_properties);
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

class vk_buffer {
public:
    vk_buffer() = default;
    ~vk_buffer() { destroy(); }

    bool create(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage) {
        destroy();
        this->device = device;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) {
            return false;
        }

        VkMemoryRequirements mem_reqs{};
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
        const uint32_t type = find_memory_type(phys, mem_reqs.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            return false;
        }

        VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = type;

        if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            return false;
        }
        if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(device, memory, nullptr);
            vkDestroyBuffer(device, buffer, nullptr);
            memory = VK_NULL_HANDLE;
            buffer = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    void destroy() {
        if (mapped && memory != VK_NULL_HANDLE) {
            vkUnmapMemory(device, memory);
            mapped = false;
        }
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }

    void * map(VkDevice device) {
        void * ptr = nullptr;
        if (vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS) {
            return nullptr;
        }
        mapped = true;
        return ptr;
    }

    void unmap(VkDevice device) {
        if (mapped) {
            vkUnmapMemory(device, memory);
            mapped = false;
        }
    }

    bool invalidate(VkDevice device, VkDeviceSize size) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = memory;
        range.offset = 0;
        range.size = size;
        return vkInvalidateMappedMemoryRanges(device, 1, &range) == VK_SUCCESS;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    bool mapped = false;
};

class vk_context {
public:
    ~vk_context() {
        if (device != VK_NULL_HANDLE) {
            if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
            if (shader != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    bool init(const std::vector<uint32_t> & shader_code) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "llama-bpe-vk";
        app.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
            return false;
        }

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (device_count == 0) {
            return false;
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        phys_device = devices[0];

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, queue_props.data());
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family = i;
                break;
            }
        }
        if (compute_queue_family == UINT32_MAX) {
            return false;
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = compute_queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;

        if (vkCreateDevice(phys_device, &dci, nullptr, &device) != VK_SUCCESS) {
            return false;
        }
        vkGetDeviceQueue(device, compute_queue_family, 0, &queue);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = compute_queue_family;
        if (vkCreateCommandPool(device, &cpci, nullptr, &command_pool) != VK_SUCCESS) {
            return false;
        }

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = shader_code.size() * sizeof(uint32_t);
        smci.pCode = shader_code.data();
        if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorSetLayoutBinding bindings[6]{};
        for (uint32_t i = 0; i < 6; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 6;
        dslci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            return false;
        }

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descriptor_set_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout) != VK_SUCCESS) {
            return false;
        }

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci2.stage = stage;
        cpci2.layout = pipeline_layout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 6;
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device, &dpci, nullptr, &descriptor_pool) != VK_SUCCESS) {
            return false;
        }

        return true;
    }

    std::optional<VkDescriptorSet> allocate_descriptor_set() {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptor_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descriptor_set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(device, &dsai, &set) != VK_SUCCESS) {
            return std::nullopt;
        }
        return set;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    uint32_t compute_queue_family = UINT32_MAX;
};

class llama_bpe_vulkan_impl : public llama_bpe_vulkan {
public:
    llama_bpe_vulkan_impl(const llama_vocab & vocab_in, int max_len_in, std::vector<merge_entry> merges_in, std::vector<uint32_t> shader_code)
        : vocab(vocab_in), max_word_len(max_len_in), merges(std::move(merges_in)) {
        ready = ctx.init(shader_code);
        if (ready) {
            ready = upload_merges();
        }
        if (ready) {
            ready = allocate_persistent();
        }
    }

    ~llama_bpe_vulkan_impl() override {
        if (ready) {
            merges_buffer.destroy();
            destroy_persistent();
        }
    }

    bool ok() const { return ready; }

    bool encode(const std::vector<std::string> & words, std::vector<llama_token> & out_tokens) override {
        if (!ready || words.empty()) {
            return false;
        }
        // Build initial symbol stream from bytes
        std::vector<word_meta> meta;
        std::vector<uint32_t> symbols;
        meta.reserve(words.size());

        for (const auto & w : words) {
            if (static_cast<int>(w.size()) > max_word_len) {
                return false;
            }
            word_meta m;
            m.offset = static_cast<uint32_t>(symbols.size());
            m.len = static_cast<uint32_t>(w.size());
            meta.push_back(m);
            for (unsigned char c : w) {
                const std::string s(1, static_cast<char>(c));
                const auto id = vocab.text_to_token(s);
                if (id == LLAMA_TOKEN_NULL) {
                    return false;
                }
                symbols.push_back(static_cast<uint32_t>(id));
            }
        }

        const size_t total_symbols = symbols.size();
        if (total_symbols == 0) {
            return false;
        }

        const VkDeviceSize meta_size = sizeof(word_meta) * meta.size();
        const VkDeviceSize sym_size = sizeof(uint32_t) * symbols.size();
        const VkDeviceSize links_size = sizeof(link_pair) * symbols.size();
        const VkDeviceSize out_size = sym_size;
        const VkDeviceSize out_meta_size = sizeof(word_meta) * meta.size();

        if (!ensure_capacity(meta_buf, meta_size) ||
            !ensure_capacity(symbols_buf, sym_size) ||
            !ensure_capacity(links_buf, links_size) ||
            !ensure_capacity(out_buf, out_size) ||
            !ensure_capacity(out_meta_buf, out_meta_size)) {
            return false;
        }
        meta_ptr = meta_buf.ptr;
        symbols_ptr = symbols_buf.ptr;
        links_ptr = links_buf.ptr;
        out_ptr = out_buf.ptr;
        out_meta_ptr = out_meta_buf.ptr;

        // upload inputs
        std::memcpy(meta_ptr, meta.data(), meta_size);
        std::memcpy(symbols_ptr, symbols.data(), sym_size);
        std::memset(links_ptr, 0xff, links_size); // initialize links to -1

        // Update descriptors in case buffers resized
        update_descriptors();

        // record command buffer
        vkResetCommandPool(ctx.device, ctx.command_pool, 0);
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &cbbi) != VK_SUCCESS) {
            return false;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        uint32_t merge_count = static_cast<uint32_t>(merges.size());
        vkCmdPushConstants(cmd, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &merge_count);
        vkCmdDispatch(cmd, static_cast<uint32_t>(meta.size()), 1, 1);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            return false;
        }

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(ctx.device, &fci, nullptr, &fence);

        if (vkQueueSubmit(ctx.queue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(ctx.device, fence, nullptr);
            return false;
        }
        if (vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(ctx.device, fence, nullptr);
            return false;
        }
        vkDestroyFence(ctx.device, fence, nullptr);

        // read back (buffers are host coherent; invalidate for safety)
        out_buf.buf.invalidate(ctx.device, out_size);
        out_meta_buf.buf.invalidate(ctx.device, out_meta_size);

        std::vector<word_meta> out_meta(meta.size());
        std::memcpy(out_meta.data(), out_meta_ptr, out_meta_size);

        std::vector<uint32_t> out_symbols(symbols.size());
        std::memcpy(out_symbols.data(), out_ptr, out_size);

        for (const auto & m : out_meta) {
            for (uint32_t i = 0; i < m.len; ++i) {
                const auto tok = out_symbols[m.offset + i];
                out_tokens.push_back(static_cast<llama_token>(tok));
            }
        }

        return true;
    }

private:
    struct mapped_buffer {
        vk_buffer buf;
        void * ptr = nullptr;
        size_t capacity = 0;
    };

    bool upload_merges() {
        merges_buffer.device = ctx.device;
        const VkDeviceSize size = sizeof(merge_entry) * merges.size();
        if (!merges_buffer.create(ctx.device, ctx.phys_device, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            return false;
        }
        void * ptr = merges_buffer.map(ctx.device);
        std::memcpy(ptr, merges.data(), size);
        merges_buffer.unmap(ctx.device);
        return true;
    }

    bool allocate_persistent() {
        // descriptor set and command buffer allocated once
        auto maybe_set = ctx.allocate_descriptor_set();
        if (!maybe_set.has_value()) {
            return false;
        }
        descriptor_set = maybe_set.value();

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(ctx.device, &cbai, &cmd) != VK_SUCCESS) {
            return false;
        }
        return true;
    }

    void destroy_persistent() {
        if (descriptor_set != VK_NULL_HANDLE) {
            vkResetDescriptorPool(ctx.device, ctx.descriptor_pool, 0);
            descriptor_set = VK_NULL_HANDLE;
        }
        if (cmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
            cmd = VK_NULL_HANDLE;
        }
        meta_buf.buf.destroy();
        symbols_buf.buf.destroy();
        links_buf.buf.destroy();
        out_buf.buf.destroy();
        out_meta_buf.buf.destroy();
    }

    bool ensure_capacity(mapped_buffer & mbuf, size_t bytes) {
        if (bytes == 0) {
            bytes = 1; // avoid zero-sized allocation
        }
        if (mbuf.capacity >= bytes) {
            return true;
        }
        mbuf.buf.destroy();
        mbuf.buf.device = ctx.device;
        size_t new_size = static_cast<size_t>(bytes * GROWTH_FACTOR);
        if (new_size < bytes) new_size = bytes;
        if (!mbuf.buf.create(ctx.device, ctx.phys_device, new_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            return false;
        }
        mbuf.ptr = mbuf.buf.map(ctx.device);
        if (mbuf.ptr == nullptr) {
            return false;
        }
        mbuf.capacity = new_size;
        descriptors_dirty = true;
        return true;
    }

    void update_descriptors() {
        if (!descriptors_dirty) {
            return;
        }
        const VkDescriptorBufferInfo merge_info{merges_buffer.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo meta_info{meta_buf.buf.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo sym_info{symbols_buf.buf.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo links_info{links_buf.buf.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo out_info{out_buf.buf.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo out_meta_info{out_meta_buf.buf.buffer, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet writes[6]{};
        const VkDescriptorBufferInfo infos[6] = {merge_info, meta_info, sym_info, links_info, out_info, out_meta_info};
        for (uint32_t i = 0; i < 6; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(ctx.device, 6, writes, 0, nullptr);
        descriptors_dirty = false;
    }

    const llama_vocab & vocab;
    int max_word_len;
    std::vector<merge_entry> merges;
    vk_context ctx;
    vk_buffer merges_buffer;
    mapped_buffer meta_buf;
    mapped_buffer symbols_buf;
    mapped_buffer links_buf;
    mapped_buffer out_buf;
    mapped_buffer out_meta_buf;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool descriptors_dirty = true;
    void * meta_ptr = nullptr;
    void * symbols_ptr = nullptr;
    void * links_ptr = nullptr;
    void * out_ptr = nullptr;
    void * out_meta_ptr = nullptr;
    bool ready = false;
};

std::vector<merge_entry> build_merge_table(const llama_vocab & vocab) {
    std::vector<merge_entry> result;
    const auto merges = vocab.get_bpe_merges();
    bool complete = true;
    result.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const auto & line = merges[rank];
        const auto pos = line.find(' ');
        if (pos == std::string::npos) {
            complete = false;
            continue;
        }
        const std::string left = line.substr(0, pos);
        const std::string right = line.substr(pos + 1);
        const std::string merged = left + right;
        const auto left_id = vocab.text_to_token(left);
        const auto right_id = vocab.text_to_token(right);
        const auto merged_id = vocab.text_to_token(merged);
        if (left_id == LLAMA_TOKEN_NULL || right_id == LLAMA_TOKEN_NULL || merged_id == LLAMA_TOKEN_NULL) {
            complete = false;
            continue;
        }
        merge_entry e;
        e.left = static_cast<uint32_t>(left_id);
        e.right = static_cast<uint32_t>(right_id);
        e.merged = static_cast<uint32_t>(merged_id);
        e.rank = static_cast<uint32_t>(rank);
        result.push_back(e);
    }

    std::sort(result.begin(), result.end(), [](const merge_entry & a, const merge_entry & b) {
        if (a.left != b.left) return a.left < b.left;
        if (a.right != b.right) return a.right < b.right;
        return a.rank < b.rank;
    });
    if (!complete) {
        result.clear();
    }
    return result;
}

} // namespace

std::unique_ptr<llama_bpe_vulkan> llama_bpe_vulkan_create(const llama_vocab & vocab, int max_word_len) {
    std::vector<uint32_t> shader_code;
#ifdef LLAMA_BPE_VK_SHADER_PATH
    const char * shader_path = LLAMA_BPE_VK_SHADER_PATH;
#else
    const char * shader_path = nullptr;
#endif
    if (shader_path == nullptr || !read_file(shader_path, shader_code)) {
        LLAMA_LOG_WARN("%s: unable to load Vulkan shader from %s\n", __func__, shader_path ? shader_path : "(null)");
        return nullptr;
    }

    auto merges = build_merge_table(vocab);
    if (merges.empty()) {
        LLAMA_LOG_WARN("%s: vocab merges could not be mapped to token ids, falling back to CPU\n", __func__);
        return nullptr;
    }

    auto impl = std::make_unique<llama_bpe_vulkan_impl>(vocab, max_word_len, std::move(merges), std::move(shader_code));
    if (!impl->ok()) {
        return nullptr;
    }
    return impl;
}

bool llama_bpe_vulkan_encode(llama_bpe_vulkan & engine, const std::vector<std::string> & words, std::vector<llama_token> & out_tokens) {
    return engine.encode(words, out_tokens);
}

#else

std::unique_ptr<llama_bpe_vulkan> llama_bpe_vulkan_create(const llama_vocab &, int) {
    return nullptr;
}

bool llama_bpe_vulkan_encode(llama_bpe_vulkan &, const std::vector<std::string> &, std::vector<llama_token> &) {
    return false;
}

#endif
