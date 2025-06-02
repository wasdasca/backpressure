#include <cstring>
#include <csignal>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <array>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <unordered_set>
#include <sys/mman.h> // 用于安全内存锁定
#include <limits>
// 调整头文件包含顺序，先包含OpenSSL相关头文件
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/provider.h>
#include <openssl/conf.h>
#include <openssl/engine.h>

// 第三方库
#include <sodium.h>
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <libbase58.h>
#include "backpressure.h"
//源码头文件

extern "C" {
 #include <openssl/sha.h>
#include "KeccakHash.h"
#include "KeccakP-1600-SnP.h"
#include "align.h"
#include "brg_endian.h"
#include "KeccakSponge.h"
#include "config.h"
#include "KeccakP-1600-times4-SnP.h"
#include "SIMD256-config.h"
}

static BackpressureController* g_backpressure_controller = nullptr;

// 内存池配置（适用于常规应用，系统内存8GB+）
constexpr size_t MEMORY_POOL_BLOCK_SIZE = 256 * 1024;  // 每个块512KB
constexpr size_t MEMORY_POOL_MAX_BLOCKS = 100;         // 最多200个块（总大小100MB）
constexpr size_t MEMORY_POOL_MAX_SIZE = MEMORY_POOL_BLOCK_SIZE * MEMORY_POOL_MAX_BLOCKS; // 100MB上限
constexpr size_t MEMORY_POOL_WARNING_SIZE = MEMORY_POOL_MAX_SIZE * 0.7; // 达到70%时警告

// 配置参数（从Config命名空间移至全局，保持与原代码结构一致）
// 根据CPU核心数动态调整线程数量
static const size_t HARDWARE_THREADS = std::thread::hardware_concurrency();
static const size_t PRODUCER_THREADS = HARDWARE_THREADS > 2 ? HARDWARE_THREADS / 2 : 1;
static const size_t CONSUMER_THREADS = HARDWARE_THREADS - PRODUCER_THREADS;
constexpr size_t MONITOR_THREADS = 1;
constexpr size_t BATCH_SIZE = 64;      // 批量处理大小
constexpr size_t MAX_QUEUE_SIZE = 10000;
// 根据实际情况调整内存池块大小
//constexpr size_t MEMORY_POOL_BLOCK_SIZE = 256 * 1024; // 256KB内存池块
//constexpr size_t CACHE_LINE_SIZE = 64; // 缓存行大小

// 内存限制配置
constexpr size_t MAX_MEMORY_USAGE = 1024 * 1024 * 1024; // 1GB
std::atomic<size_t> current_memory_usage{0};

// BIP44路径配置（波场标准）
constexpr uint32_t BIP44_PURPOSE = 0x8000002C;   // 44'
constexpr uint32_t BIP44_COIN_TYPE = 0x800000C3; // 195' (TRX)
constexpr uint32_t BIP44_ACCOUNT = 0x80000000;   // 0'
constexpr uint32_t BIP44_CHANGE = 0x00000000;    // 外部链
constexpr uint32_t BIP44_START_INDEX = 0;        // 起始索引
constexpr size_t DERIVED_KEYS_PER_SEED = 5;      // 每个种子派生5个密钥（1根+4子）

// BIP32扩展密钥结构
struct BIP32Key {
    uint32_t version;       // 版本号（xprv: 0x0488ADE4, xpub: 0x0488B21E）
    uint8_t depth;          // 深度
    uint32_t parent_fingerprint; // 父密钥指纹
    uint32_t child_number;  // 子密钥编号
    uint8_t chain_code[32]; // 链码
    uint8_t private_key[32]; // 私钥（可选）
    uint8_t public_key[65];  // 公钥（可选，未压缩格式）
    bool is_private;        // 是否包含私钥

    BIP32Key() : version(0), depth(0), parent_fingerprint(0), 
                child_number(0), is_private(false) {
        memset(chain_code, 0, 32);
        memset(private_key, 0, 32);
        memset(public_key, 0, 65);
    }
};

// 内存对齐分配器
template <typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;

    template <typename U, size_t OtherAlignment>
    AlignedAllocator(const AlignedAllocator<U, OtherAlignment>&) {}

    // 替换AlignedAllocator中的allocate和deallocate
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        void* ptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) {
        free(p);
    }
};

// 添加Block结构体定义
struct Block {
    void* memory;
    size_t size;
    size_t used;
    Block* next;
    
    Block() : memory(nullptr), size(0), used(0), next(nullptr) {}
};

// 修复1: 改进ThreadLocalMemoryPool的内存管理（优化版）
class ImprovedThreadLocalMemoryPool {
private:
    struct LargeAllocation {
        void* ptr;
        size_t size;
        LargeAllocation* next;
    };
    
    Block* current_block;
    LargeAllocation* large_allocations; // 跟踪大内存分配
    const size_t block_size;
    std::atomic<size_t> total_allocated{0};
    
public:
    explicit ImprovedThreadLocalMemoryPool(size_t size = MEMORY_POOL_BLOCK_SIZE)
        : block_size(size), large_allocations(nullptr) {
        current_block = allocate_block();
    }
    
    ~ImprovedThreadLocalMemoryPool() {
        cleanup_large_allocations();
        
        while (current_block) {
            Block* next = current_block->next;
            sodium_memzero(current_block->memory, current_block->size);
            free(current_block->memory);
            
            if (g_backpressure_controller) {
                g_backpressure_controller->sub_memory_usage(current_block->size);
            }
            
            free(current_block);
            current_block = next;
        }
    }

    void* allocate(size_t size) {
        // 1. 检查背压状态（提前检查，避免无效操作）
        if (g_backpressure_controller) {
            auto decision = g_backpressure_controller->check_backpressure();
            if (decision.level >= BackpressureController::BackpressureLevel::WARNING) {
                cleanup_large_allocations(); // 紧急清理
                if (decision.level == BackpressureController::BackpressureLevel::EMERGENCY) {
                    throw std::bad_alloc();
                }
            }
        }

        // 2. 按需对齐（优化对齐策略）
        size_t aligned_size = (size >= 64 || size % 64 == 0) ? size : ((size / 64) + 1) * 64;

        // 3. 大内存分配
        if (aligned_size > block_size / 2) {
            void* ptr = aligned_alloc(64, aligned_size);
            if (!ptr) throw std::bad_alloc();

            // 原子更新统计
            size_t new_total = total_allocated.fetch_add(aligned_size) + aligned_size;
            bool should_rollback = false;
            
            if (g_backpressure_controller) {
                g_backpressure_controller->add_memory_usage(aligned_size);
                should_rollback = (new_total > MEMORY_POOL_MAX_SIZE);
            }

            if (should_rollback) {
                total_allocated.fetch_sub(aligned_size);
                free(ptr);
                throw std::bad_alloc();
            }

            // 使用malloc而非new（避免二次分配）
            LargeAllocation* alloc = static_cast<LargeAllocation*>(malloc(sizeof(LargeAllocation)));
            if (!alloc) {
                total_allocated.fetch_sub(aligned_size);
                free(ptr);
                throw std::bad_alloc();
            }
            *alloc = {ptr, aligned_size, large_allocations};
            large_allocations = alloc;
            return ptr;
        }

        // 4. 小内存分配：遍历所有块寻找合适位置
        Block* suitable_block = nullptr;
        Block* block = current_block;
        while (block) {
            if (block->used + aligned_size <= block->size) {
                suitable_block = block;
                break;
            }
            block = block->next;
        }

        // 5. 找到可用块
        if (suitable_block) {
            void* ptr = static_cast<char*>(suitable_block->memory) + suitable_block->used;
            suitable_block->used += aligned_size;
            total_allocated.fetch_add(aligned_size); // 更新统计
            return ptr;
        }

        // 6. 分配新块（非递归实现）
        Block* new_block = allocate_block();
        if (!new_block) throw std::bad_alloc();

        // 初始化新块并返回内存
        new_block->used = aligned_size;
        new_block->next = current_block;
        current_block = new_block;
        total_allocated.fetch_add(aligned_size); // 更新统计
        
        return static_cast<char*>(new_block->memory);
    }

    // 新增：定期清理（可由外部调用）
    void periodic_cleanup() {
        cleanup_large_allocations();
        // 可选：合并部分小块
    }

    // 释放大内存分配
    void cleanup_large_allocations() {
        LargeAllocation* current = large_allocations;
        while (current) {
            LargeAllocation* next = current->next;
            sodium_memzero(current->ptr, current->size);
            free(current->ptr);
            
            total_allocated.fetch_sub(current->size);
            if (g_backpressure_controller) {
                g_backpressure_controller->sub_memory_usage(current->size);
            }
            
            free(current);
            current = next;
        }
        large_allocations = nullptr;
    }
    
    // 获取当前分配的内存总量
    size_t get_allocated_size() const {
        return total_allocated.load();
    }

private:
    Block* allocate_block() {
        void* memory = malloc(block_size);
        if (!memory) throw std::bad_alloc();

        Block* block = static_cast<Block*>(malloc(sizeof(Block)));
        if (!block) {
            free(memory);
            throw std::bad_alloc();
        }

        block->memory = memory;
        block->size = block_size;
        block->used = 0;
        block->next = nullptr;
        
        if (g_backpressure_controller) {
            g_backpressure_controller->add_memory_usage(block_size);
        }
        
        return block;
    }
};    

// 修复2: 改进BatchTask的内存管理
struct ImprovedBatchTask {
    std::vector<std::vector<uint8_t>> private_keys;
    std::vector<std::vector<uint8_t>> public_keys;
    std::vector<std::string> addresses;
    std::chrono::steady_clock::time_point creation_time;
    
    ImprovedBatchTask() {
        creation_time = std::chrono::steady_clock::now();
        private_keys.reserve(BATCH_SIZE * DERIVED_KEYS_PER_SEED);
        public_keys.reserve(BATCH_SIZE * DERIVED_KEYS_PER_SEED);
        addresses.reserve(BATCH_SIZE * DERIVED_KEYS_PER_SEED);
    }
    
    ~ImprovedBatchTask() {
        // 安全清除所有私钥
        for (auto& key : private_keys) {
            if (!key.empty()) {
                sodium_memzero(key.data(), key.size());
            }
        }
    }
    
    // 获取任务存活时间
    double get_age_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - creation_time).count();
    }
    
    // 估算内存使用量
    size_t estimate_memory_usage() const {
        size_t total = 0;
        for (const auto& key : private_keys) {
            total += key.capacity();
        }
        for (const auto& key : public_keys) {
            total += key.capacity();
        }
        for (const auto& addr : addresses) {
            total += addr.capacity();
        }
        return total;
    }
};

// 全局状态(按缓存行对齐避免伪共享)
struct alignas(CACHE_LINE_SIZE) GlobalState {
    std::atomic<bool> running{true};
    std::atomic<uint64_t> keys_generated{0}; 
    std::atomic<uint64_t> addresses_checked{0}; 
    std::atomic<uint64_t> matches_found{0}; 
    std::unordered_set<std::string> target_addresses; // 存储所有目标地址
    std::string result_file;
    mutable std::mutex file_mutex; // 保护文件操作的互斥锁
} global_state;

// 信号处理函数（添加线程ID输出）
void signal_handler(int signum) {
    global_state.running = false;
    std::cout << "\nThread " << std::this_thread::get_id() << " received signal " << signum 
              << ", initiating shutdown...\n";
}

// 私钥有效性验证
bool is_valid_private_key(const uint8_t* private_key) {
    // 曲线最大私钥: FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAE DCE6 AF48 A03B BF25 E8CD 0364 1411
    static const uint8_t max_privkey[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };

    // 检查是否全0
    for (int i = 0; i < 32; i++) {
        if (private_key[i] != 0) {
            break;
        }
        if (i == 31) return false; // 全0无效
    }

    // 按字节比较是否小于max_privkey
    for (int i = 0; i < 32; i++) {
        if (private_key[i] < max_privkey[i]) return true;
        if (private_key[i] > max_privkey[i]) return false;
    }
    return false; // 等于max_privkey也无效
}

// 使用libsecp256k1的私钥转公钥实现
bool private_to_public(const uint8_t* private_key, uint8_t* public_key) {
    // 验证私钥有效性
    if (!is_valid_private_key(private_key)) {
        return false;
    }

    // 创建secp256k1上下文
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        return false;
    }

    secp256k1_pubkey pubkey;
    int ret = secp256k1_ec_pubkey_create(ctx, &pubkey, private_key);

    if (ret) {
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, public_key, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);
    }

    secp256k1_context_destroy(ctx);
    return ret != 0;
}

// 从私钥生成公钥(优化接口)
std::vector<uint8_t> private_to_public_avx2(const std::vector<uint8_t>& private_key) {
    std::vector<uint8_t> public_key(65);
    if (!private_to_public(private_key.data(), public_key.data())) {
        throw std::runtime_error("Invalid private key or failed to generate public key");
    }
    return public_key;
}

// 使用KeccakHash.c实现SHA-3-256
std::vector<uint8_t> sha3_256(const std::vector<uint8_t>& data) {
    Keccak_HashInstance instance;
    std::vector<uint8_t> hash(32);  // SHA-3-256 输出32字节

    // 初始化SHA-3-256 (rate=1344, capacity=256, 输出长度=256位)
    if (Keccak_HashInitialize(&instance, 1344, 256, 256, 0x06) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 initialization failed");
    }

    // 更新哈希数据
    if (Keccak_HashUpdate(&instance, data.data(), data.size() * 8) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 update failed");
    }

    // 完成哈希计算
    if (Keccak_HashFinal(&instance, hash.data()) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 finalization failed");
    }

    return hash;
}

std::string public_to_tron(const std::vector<uint8_t>& public_key) {
    if (public_key.size() != 65 || public_key[0] != 0x04) {
        throw std::invalid_argument("TRON address generation requires uncompressed public key (65 bytes starting with 0x04)");
    }

    std::vector<uint8_t> sha3_digest = sha3_256(
        std::vector<uint8_t>(public_key.begin() + 1, public_key.end()));

    std::vector<uint8_t> address_hash(sha3_digest.end() - 20, sha3_digest.end());

    std::vector<char> encoded(50, 0);
    size_t encoded_len = encoded.size();

    // 使用 version = 0x41，data 只传 20 字节的 payload
    if (!b58check_enc(encoded.data(), &encoded_len, 
                      0x41,  // version 字节
                      address_hash.data(), address_hash.size())) {
        throw std::runtime_error("Base58Check encoding failed");
    }

    return std::string(encoded.data(), encoded_len);
}


// 自定义HMAC - SHA512计算函数
std::vector<uint8_t> hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len) {
    std::vector<uint8_t> result(64);
    unsigned int len = 64;
    if (HMAC(EVP_sha512(), key, key_len, data, data_len, result.data(), &len) == nullptr) {
        throw std::runtime_error("HMAC-SHA512 failed: " + std::string(ERR_error_string(ERR_get_error(), NULL)));
    }
    return result;
}

// 从种子生成主密钥
BIP32Key generate_master_key(const std::vector<uint8_t>& seed) {
    BIP32Key master_key;

    // 计算HMAC-SHA512("Bitcoin seed", seed)
    auto hmac_result = hmac_sha512(reinterpret_cast<const uint8_t*>("Bitcoin seed"), 12, seed.data(), seed.size());

    // 前32字节作为私钥
    memcpy(master_key.private_key, hmac_result.data(), 32);

    // 验证私钥有效性
    if (!is_valid_private_key(master_key.private_key)) {
        throw std::runtime_error("Generated master key is invalid");
    }

    // 后32字节作为链码
    memcpy(master_key.chain_code, hmac_result.data() + 32, 32);

    // 生成对应的公钥
    if (!private_to_public(master_key.private_key, master_key.public_key)) {
        throw std::runtime_error("Failed to generate public key for master key");
    }

    // 设置主密钥的其他参数
    master_key.version = 0x0488ADE4; // xprv版本
    master_key.depth = 0;
    master_key.parent_fingerprint = 0;
    master_key.child_number = 0;
    master_key.is_private = true;

    return master_key;
}

// 计算密钥指纹（前4字节的公钥哈希）
uint32_t calculate_key_fingerprint(const uint8_t* public_key) {
    // 1. 对公钥进行SHA-256哈希
    uint8_t sha256_digest[32];
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }

    const EVP_MD* md = EVP_sha256();
    if (!md) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHA256 not available");
    }

    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, public_key, 65) != 1 ||
        EVP_DigestFinal_ex(ctx, sha256_digest, nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHA256 computation failed");
    }

    EVP_MD_CTX_free(ctx);

    // 2. 对SHA-256结果进行RIPEMD160哈希
    uint8_t ripemd160_digest[20];
    ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }

    md = EVP_ripemd160();
    if (!md) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("RIPEMD160 not available");
    }

    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, sha256_digest, 32) != 1 ||
        EVP_DigestFinal_ex(ctx, ripemd160_digest, nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("RIPEMD160 computation failed");
    }

    EVP_MD_CTX_free(ctx);

    // 返回前4字节作为指纹
    return (ripemd160_digest[0] << 24) | 
           (ripemd160_digest[1] << 16) | 
           (ripemd160_digest[2] << 8) | 
            ripemd160_digest[3];
}

// 修复：确保data数组足够大（非硬化派生需要69字节）
BIP32Key derive_child_key(const BIP32Key& parent_key, uint32_t child_index) {
    BIP32Key child_key;
    // 检查是否为硬化派生（index >= 0x80000000）
    bool is_hardened = (child_index & 0x80000000) != 0;
    uint8_t data[69]; // 足够存储非硬化派生数据（65字节公钥+4字节索引）

    // 准备HMAC输入数据
    if (is_hardened) {
        // 硬化派生：0x00 || 父私钥 || 子索引
        data[0] = 0x00;
        memcpy(data + 1, parent_key.private_key, 32);
        memcpy(data + 33, &child_index, 4);
    } else {
        // 非硬化派生：父公钥 || 子索引
        memcpy(data, parent_key.public_key, 65);
        memcpy(data + 65, &child_index, 4);
    }

    // 计算HMAC-SHA512，使用父密钥的chain_code作为密钥
    auto hmac_result = hmac_sha512(parent_key.chain_code, 32, data, is_hardened ? 37 : 69);

    // 分解HMAC结果为子私钥和子链码
    // 子私钥 = (父私钥 + IL) mod n
    // 子链码 = IR
    // 检查IL是否小于曲线阶n
    static const uint8_t curve_n[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    bool il_less_than_n = true;
    for (int i = 0; i < 32; i++) {
        if (hmac_result[i] < curve_n[i]) break;
        if (hmac_result[i] > curve_n[i]) {
            il_less_than_n = false;
            break;
        }
    }
    if (!il_less_than_n) {
        throw std::runtime_error("Derived private key is invalid (IL >= n)");
    }

    // 创建secp256k1上下文用于密钥加法
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) {
        throw std::runtime_error("Failed to create secp256k1 context");
    }

    // 计算子私钥 = 父私钥 + IL
    if (parent_key.is_private) {
        uint8_t child_private_key[32];
        memcpy(child_private_key, parent_key.private_key, 32);
        // 执行私钥加法：child_private_key = parent_private_key + hmac_result
        if (!secp256k1_ec_seckey_tweak_add(ctx, child_private_key, hmac_result.data())) {
            secp256k1_context_destroy(ctx);
            throw std::runtime_error("Failed to add private keys");
        }
        memcpy(child_key.private_key, child_private_key, 32);
        // 生成对应的公钥
        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, child_private_key)) {
            secp256k1_context_destroy(ctx);
            throw std::runtime_error("Failed to generate public key for child key");
        }
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, child_key.public_key, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);
        child_key.is_private = true;
    } else {
        // 非硬化派生的公钥派生
        secp256k1_pubkey parent_pubkey;
        if (!secp256k1_ec_pubkey_parse(ctx, &parent_pubkey, parent_key.public_key, 65)) {
            secp256k1_context_destroy(ctx);
            throw std::runtime_error("Failed to parse parent public key");
        }
        // 调整公钥：child_pubkey = parent_pubkey + IL*G
        if (!secp256k1_ec_pubkey_tweak_add(ctx, &parent_pubkey, hmac_result.data())) {
            secp256k1_context_destroy(ctx);
            throw std::runtime_error("Failed to derive child public key");
        }
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, child_key.public_key, &pubkey_len, &parent_pubkey, SECP256K1_EC_UNCOMPRESSED);
        child_key.is_private = false;
    }
    secp256k1_context_destroy(ctx);

    // 设置子链码
    memcpy(child_key.chain_code, hmac_result.data() + 32, 32);

    // 设置子密钥的其他参数
    child_key.version = parent_key.version;
    child_key.depth = parent_key.depth + 1;
    child_key.parent_fingerprint = calculate_key_fingerprint(parent_key.public_key);
    child_key.child_number = child_index;

    return child_key;
}

// 从种子派生HD密钥（生成1根+4子密钥）
std::vector<std::vector<uint8_t>> derive_hd_keys(const std::vector<uint8_t>& seed) {
    std::vector<std::vector<uint8_t>> keys;
    
    try {
        // 生成主密钥
        BIP32Key master_key = generate_master_key(seed);
        
        // 存储根密钥
        std::vector<uint8_t> root_private_key(32);
        memcpy(root_private_key.data(), master_key.private_key, 32);
        keys.push_back(root_private_key);
        
        // 派生路径基础部分: m/44'/195'/0'/0
        BIP32Key derived_key = master_key;
        
        // 硬化派生: 44'
        derived_key = derive_child_key(derived_key, BIP44_PURPOSE | 0x80000000);
        
        // 硬化派生: 195' (TRX)
        derived_key = derive_child_key(derived_key, BIP44_COIN_TYPE | 0x80000000);
        
        // 硬化派生: 0' (账户)
        derived_key = derive_child_key(derived_key, BIP44_ACCOUNT | 0x80000000);
        
        // 非硬化派生: 0 (外部链)
        derived_key = derive_child_key(derived_key, BIP44_CHANGE);
        
        // 派生4个子密钥 (索引0-3)
        for (uint32_t i = 0; i < 4; ++i) {
            BIP32Key child_key = derive_child_key(derived_key, i);
            
            // 提取子私钥
            std::vector<uint8_t> child_private_key(32);
            memcpy(child_private_key.data(), child_key.private_key, 32);
            keys.push_back(child_private_key);
        }
    } catch (const std::exception& e) {
        std::cerr << "HD key derivation error: " << e.what() << std::endl;
        // 安全清理已生成的部分密钥
        for (auto& key : keys) {
            sodium_memzero(key.data(), key.size());
        }
        keys.clear();
        throw; // 重新抛出异常，由调用者处理
    }
    
    return keys;
}

// 从熵生成种子（使用libsodium替代libbtc的sha256，避免冲突）
std::vector<uint8_t> generate_seed() {
    std::vector<uint8_t> entropy(32);
    randombytes_buf(entropy.data(), 32);  // 直接调用，无需检查返回值
    
    std::vector<uint8_t> seed(64);
    crypto_generichash(seed.data(), seed.size(), entropy.data(), entropy.size(), NULL, 0);
    sodium_memzero(entropy.data(), entropy.size());
    return seed;
}

// AVX2优化的批量密钥生成（生成1根+4子密钥）
class AVX2KeyGenerator {
private:
    struct ThreadLocalState {
        std::vector<uint8_t> seed_buffer;
        secp256k1_context* secp_context;
        
        ThreadLocalState() : secp_context(secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY)) {
            seed_buffer.resize(64);
        }
        
        ~ThreadLocalState() {
            if (secp_context) {
                secp256k1_context_destroy(secp_context);
            }
            sodium_memzero(seed_buffer.data(), seed_buffer.size());
        }
    };
    
    static thread_local std::unique_ptr<ThreadLocalState> tls_state;
    
public:
    static void generate_batch(ImprovedBatchTask& batch, ImprovedThreadLocalMemoryPool& pool) {
        // 初始化线程本地状态
        if (!tls_state) {
            tls_state = std::make_unique<ThreadLocalState>();
        }
        
        size_t keys_added = 0;
        
        while (keys_added < BATCH_SIZE * DERIVED_KEYS_PER_SEED) {
            try {
                // 生成种子
                auto seed = generate_seed();
                
                // 派生HD密钥（1根+4子）
                auto derived_keys = derive_hd_keys(seed);
                
                // 安全擦除种子
                sodium_memzero(seed.data(), seed.size());
                
                // 存储到批量任务
                for (const auto& key : derived_keys) {
                    if (keys_added >= BATCH_SIZE * DERIVED_KEYS_PER_SEED) break;
                    
                    // 生成对应的公钥
                    auto public_key = private_to_public_avx2(key);
                    
                    // 存储私钥和公钥
                    batch.private_keys.push_back(std::move(key));
                    batch.public_keys.push_back(std::move(public_key));
                    
                    keys_added++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Key generation error: " << e.what() << " - using fallback\n";
                
                // 生成随机密钥作为fallback
                std::vector<uint8_t> private_key(32);
                do {
                    randombytes_buf(private_key.data(), 32);
                } while (!is_valid_private_key(private_key.data()));
                
                // 生成公钥
                auto public_key = private_to_public_avx2(private_key);
                
                // 存储结果
                if (keys_added < BATCH_SIZE * DERIVED_KEYS_PER_SEED) {
                    batch.private_keys.push_back(std::move(private_key));
                    batch.public_keys.push_back(std::move(public_key));
                    keys_added++;
                }
            }
        }
        
        // 更新统计（注意每个种子生成5个密钥）
        global_state.keys_generated.fetch_add(keys_added, std::memory_order_relaxed);
    }
    
    static void thread_cleanup() {
        tls_state.reset();
    }
};

// 初始化线程本地变量
thread_local std::unique_ptr<AVX2KeyGenerator::ThreadLocalState> AVX2KeyGenerator::tls_state;

// 公钥有效性验证
bool is_valid_public_key(const std::vector<uint8_t>& key) {
    return key.size() == 65 && key[0] == 0x04;
}

// AVX2优化的地址生成
class AVX2AddressGenerator {
public:
    static void process_batch(ImprovedBatchTask& batch) {
        for (size_t i = 0; i < batch.private_keys.size(); ++i) {
            try {
                // 确保每个public_key都有效
                const auto& private_key = batch.private_keys[i];
                auto public_key = private_to_public_avx2(private_key);
                
                if (!is_valid_public_key(public_key)) {
                    throw std::runtime_error("Invalid public key format");
                }
                
                batch.public_keys[i] = std::move(public_key); // 确保写入public_key
                
                // 从公钥生成地址
                auto address = public_to_tron(batch.public_keys[i]);
                
                // 存储地址
                batch.addresses.push_back(std::move(address));
                
                // 更新统计
                global_state.addresses_checked.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                std::cerr << "Address generation error: " << e.what() << std::endl;
                
                // 生成随机地址作为fallback
                std::vector<uint8_t> random_private(32);
                do {
                    randombytes_buf(random_private.data(), 32);
                } while (!is_valid_private_key(random_private.data()));
                
                auto public_key = private_to_public_avx2(random_private);
                
                // 确保fallback的public_key有效
                if (!is_valid_public_key(public_key)) {
                    std::cerr << "Fallback public key is still invalid, this should not happen\n";
                    continue;
                }
                
                // 存储fallback结果，确保所有字段都被正确填充
                batch.private_keys[i] = std::move(random_private);
                batch.public_keys[i] = std::move(public_key);
                
                // 再次尝试生成地址
                auto address = public_to_tron(batch.public_keys[i]);
                batch.addresses[i] = std::move(address);
                
                // 更新统计
                global_state.addresses_checked.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
};

// 修复3: 改进生产者线程，添加背压控制
void improved_producer_thread(MonitoredQueue<std::shared_ptr<ImprovedBatchTask>>& task_queue) {
    ImprovedThreadLocalMemoryPool memory_pool;
    
    while (global_state.running) {
        try {
            // 检查背压状态
            if (g_backpressure_controller) {
                auto decision = g_backpressure_controller->check_backpressure();
                if (decision.should_throttle) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(decision.delay_ms));
                    
                    if (decision.level >= BackpressureController::BackpressureLevel::CRITICAL) {
                        std::cout << "Producer throttled: " << decision.reason << std::endl;
                        continue; // 跳过这次生产
                    }
                }
            }
            
            // 创建新批次任务
            auto batch = std::make_shared<ImprovedBatchTask>();
            
            // 生成密钥 - 这里使用你原有的逻辑
            AVX2KeyGenerator::generate_batch(*batch, memory_pool);
            
            // 尝试加入队列，如果失败则等待
            int retry_count = 0;
            while (!task_queue.enqueue(batch) && global_state.running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                retry_count++;
                
                // 如果重试太多次，说明消费跟不上，需要减慢生产
                if (retry_count > 20) {
                    std::cout << "Producer backing off due to queue pressure" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    break;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Producer error: " << e.what() << std::endl;
            // 发生异常时暂停一下
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    // 清理线程本地资源
    AVX2KeyGenerator::thread_cleanup();
}

// 修复4: 改进消费者线程
void improved_consumer_thread(MonitoredQueue<std::shared_ptr<ImprovedBatchTask>>& task_queue) {
    while (global_state.running) {
        std::shared_ptr<ImprovedBatchTask> batch;
        
        if (!task_queue.dequeue(batch)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        try {
            if (batch->get_age_seconds() > 30.0) {
                std::cout << "Dropping expired batch task" << std::endl;
                batch->addresses.clear();
                batch->private_keys.clear();
                batch->addresses.shrink_to_fit();
                batch->private_keys.shrink_to_fit();
                batch.reset(); // 显式释放引用
                continue;
            }

            AVX2AddressGenerator::process_batch(*batch);

            for (size_t i = 0; i < batch->addresses.size(); ++i) {
                if (global_state.target_addresses.count(batch->addresses[i]) > 0) {
                    global_state.matches_found.fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(global_state.file_mutex);

                    std::ofstream outfile(global_state.result_file, std::ios::app);
                    if (outfile.is_open()) {
                        std::ostringstream ss;
                        ss << "Match found: " << batch->addresses[i] << "\n";
                        ss << "Private Key: ";
                        for (uint8_t byte : batch->private_keys[i]) {
                            ss << std::hex << std::setw(2) << std::setfill('0')
                               << static_cast<int>(byte);
                        }
                        ss << "\n\n";
                        outfile << ss.str();
                    }
                }
            }

            // 清理 batch 内容并释放内存
            batch->addresses.clear();
            batch->private_keys.clear();
            batch->addresses.shrink_to_fit();
            batch->private_keys.shrink_to_fit();
            batch.reset();

        } catch (const std::exception& e) {
            std::cerr << "Consumer error: " << e.what() << std::endl;
        }
    }
}


// 修复5: 改进内存监控线程
void improved_memory_monitor_thread() {
    auto last_report_time = std::chrono::steady_clock::now();
    
    while (global_state.running) {
        if (g_backpressure_controller) {
            auto stats = g_backpressure_controller->get_statistics();
            auto now = std::chrono::steady_clock::now();
            
            // 每30秒报告一次详细状态
            if (std::chrono::duration<double>(now - last_report_time).count() > 30.0) {
                std::cout << "\n=== Memory Status ===" << std::endl;
                std::cout << "Current Memory: " << stats.current_memory_mb << " MB" << std::endl;
                std::cout << "Peak Memory: " << stats.peak_memory_mb << " MB" << std::endl;
                std::cout << "Queue Size: " << stats.current_queue_size << "/" << stats.max_queue_size << std::endl;
                std::cout << "Backpressure Level: " << static_cast<int>(stats.current_level) << std::endl;
                std::cout << "Backpressure Triggers: " << stats.backpressure_triggers << std::endl;
                std::cout << "Emergency Stops: " << stats.emergency_stops << std::endl;
                std::cout << "MEMORY_POOL_MAX_SIZE: " << MEMORY_POOL_MAX_SIZE << std::endl;
                std::cout << "PRODUCER_THREADS: " << PRODUCER_THREADS << std::endl;
                std::cout << "===================" << std::endl;
                
                last_report_time = now;
            }

            // 如果处于紧急状态，尝试清理
            if (stats.current_level >= BackpressureController::BackpressureLevel::CRITICAL) {
                std::cout << "Memory pressure detected, triggering cleanup..." << std::endl;
                // 这里可以触发额外的清理操作
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

// 监控线程函数
void monitor_thread() {
    uint64_t last_checked = 0;
    uint64_t last_matches = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (global_state.running) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        int elapsed_int = static_cast<int>(elapsed);
        int hours = elapsed_int / 3600;
        int minutes = (elapsed_int % 3600) / 60;
        int seconds = elapsed_int % 60;
        
        double speed = (global_state.addresses_checked.load() - last_checked) / 5.0;
        std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << speed << " addr/s | "
                  << "Total: " << global_state.addresses_checked.load() << " | "
                  << "Matches: " << global_state.matches_found.load() << " | "
                  << "Uptime: " << hours << "h " << minutes << "m " << seconds << "s"
                  << std::flush;
        
        last_checked = global_state.addresses_checked.load();
        last_matches = global_state.matches_found.load();
    }
    std::cout << std::endl;
}

// 加载目标地址
bool load_target_addresses(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open target addresses file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 去除空白字符
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        
        if (!line.empty()) {
            global_state.target_addresses.insert(line);
        }
    }
    
    file.close();
    std::cout << "Loaded " << global_state.target_addresses.size() << " target addresses" << std::endl;
    return true;
}

bool my_sha256(void* digest, const void* data, size_t len) {
    return SHA256(static_cast<const unsigned char*>(data), len, static_cast<unsigned char*>(digest)) != nullptr;
}

// 清理OpenSSL资源
void cleanup_openssl() {
    EVP_cleanup();
    ERR_free_strings();
    CRYPTO_cleanup_all_ex_data();
    CONF_modules_unload(1);
    CONF_modules_free();
    ENGINE_cleanup();
}

// 主函数
int main(int argc, char* argv[]) {

    // ⭐️ 关键修复：设置 Base58 的 SHA256 实现
    b58_sha256_impl = my_sha256;
    // 检查是否设置成功
    if (!b58_sha256_impl) {
        std::cerr << "Error: SHA256 implementation not set for Base58!" << std::endl;
        return 1;
    }

    // 初始化OpenSSL
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
    
    // 初始化libsodium
    if (sodium_init() < 0) {
        std::cerr << "Failed to initialize libsodium" << std::endl;
        return 1;
    }
    
    // 初始化secp256k1
    secp256k1_context* secp_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!secp_ctx) {
        std::cerr << "Failed to initialize secp256k1" << std::endl;
        return 1;
    }
    secp256k1_context_destroy(secp_ctx);

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 命令行参数解析
    std::string target_file = "targets.txt";
    global_state.result_file = "matches.txt";
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t" && i + 1 < argc) {
            target_file = argv[++i];
        } else if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            global_state.result_file = argv[++i];
        } else if (std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [-t targets_file] [-o output_file]\n";
            return 0;
        }
    }
    
    // 加载目标地址
    if (!load_target_addresses(target_file)) {
        return 1;
    }

    // 初始化背压控制器
    g_backpressure_controller = new BackpressureController(
        500,    // 最大队列大小
        4096,    // 内存警告阈值6GB
        6144     // 内存临界阈值7.5GB
    );
    
    // 创建监控队列
    MonitoredQueue<std::shared_ptr<ImprovedBatchTask>> task_queue(MAX_QUEUE_SIZE, g_backpressure_controller);
    
    // 创建并启动线程
    std::vector<std::thread> producer_threads;
    std::vector<std::thread> consumer_threads;
    std::vector<std::thread> monitor_threads;
    
    // 启动生产者线程
    for (size_t i = 0; i < PRODUCER_THREADS; ++i) {
        producer_threads.emplace_back(improved_producer_thread, std::ref(task_queue));
    }
    
    // 启动消费者线程
    for (size_t i = 0; i < CONSUMER_THREADS; ++i) {
        consumer_threads.emplace_back(improved_consumer_thread, std::ref(task_queue));
    }
    
    // 创建监控线程
    monitor_threads.emplace_back(monitor_thread);
    
    // 添加内存监控线程
    monitor_threads.emplace_back(improved_memory_monitor_thread);
    
    // 等待线程结束
    for (auto& t : producer_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // 清空队列
    while (!task_queue.empty()) {
        std::shared_ptr<ImprovedBatchTask> batch;
        if (task_queue.dequeue(batch)) {
            // 安全清理私钥
            for (auto& key : batch->private_keys) {
                sodium_memzero(key.data(), key.size());
            }
        }
    }
    
    // 停止消费者和监控线程
    global_state.running = false;
    
    for (auto& t : consumer_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    for (auto& t : monitor_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // 清理资源
    delete g_backpressure_controller;
    g_backpressure_controller = nullptr;
    cleanup_openssl();
    
    std::cout << "Program terminated. Total matches found: " << global_state.matches_found.load() << std::endl;
    
    return 0;
} 
