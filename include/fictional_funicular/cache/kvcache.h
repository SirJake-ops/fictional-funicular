//
// Created by jake on 3/20/26.
//

#ifndef LLM_INFERENCE_ENGINE_KVCACHE_H
#define LLM_INFERENCE_ENGINE_KVCACHE_H
#include "onnxruntime_cxx_api.h"


namespace cache {
    struct KVCache {
        std::vector<Ort::Value> _keys_cache;
        std::vector<Ort::Value> _values_cache;

        void clear() {
            _keys_cache.clear();
            _values_cache.clear();
        }
    };
}

#endif //LLM_INFERENCE_ENGINE_KVCACHE_H
