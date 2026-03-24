//
// Created by jake on 3/20/26.
//

#ifndef LLM_INFERENCE_ENGINE_KVCACHE_H
#define LLM_INFERENCE_ENGINE_KVCACHE_H
#include "onnxruntime_cxx_api.h"


namespace cache {
    struct KVCache {
        std::vector<Ort::Value> _keys;
        std::vector<Ort::Value> _values;

        void clear() {
            _keys.clear();
            _values.clear();
        }
    };
}

#endif //LLM_INFERENCE_ENGINE_KVCACHE_H
