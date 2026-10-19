#include "qwen3_asr.h"
#include "language_prompt.h"
#include "timing.h"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <fstream>
#include <ggml-impl.h>

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }

    if (!decoder_.load_model(model_path)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }

    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);

    model_loaded_ = true;
    int64_t t_end = get_time_ms();

    GGML_LOG_INFO("%s: Qwen3ASR loaded in %lld ms\n", __func__, (long long)(t_end - t_start));
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return transcribe_internal(samples.data(), samples.size(), params);
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}


// Resolve a user-supplied language (canonical name, lowercase name or ISO code)
// to its prompt entry; nullptr means "let the model detect the language".
static const language_prompt_entry * find_language_prompt(const std::string & language) {
    std::string key;
    for (unsigned char c : language) {
        if (!std::isspace(c)) key.push_back(static_cast<char>(std::tolower(c)));
    }
    if (key.empty()) {
        return nullptr;
    }
    for (const auto & entry : LANGUAGE_PROMPT_TABLE) {
        for (const char * alias : entry.aliases) {
            if (key == alias) {
                return &entry;
            }
        }
    }
    return nullptr;
}

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();

    encoder_.set_n_threads(params.n_threads);
    decoder_.set_n_threads(params.n_threads);
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Mel spectrogram: [%d, %d]\n", __func__, mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    // Debug aid (see QWEN3_ASR_DEBUG_HIDDEN in text_decoder.cpp): also dump the
    // mel spectrogram and the encoder output of this call.
    if (const char * dbg_dir = getenv("QWEN3_ASR_DEBUG_HIDDEN")) {
        std::ofstream fm(std::string(dbg_dir) + "/mel.f32", std::ios::binary);
        fm.write(reinterpret_cast<const char *>(mel.data.data()), mel.data.size() * sizeof(float));
        std::ofstream fa(std::string(dbg_dir) + "/audio_features.f32", std::ios::binary);
        fa.write(reinterpret_cast<const char *>(audio_features.data()), audio_features.size() * sizeof(float));
        fprintf(stderr, "dumped mel [%d x %d] and audio features [%d x %d]\n", mel.n_mel, mel.n_len, n_audio_frames, text_hparams.hidden_size);
    }
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Audio features: [%d, %d]\n", __func__, n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.language);
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Input tokens: %zu\n", __func__, input_tokens.size());
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    const bool decoded = params.beam_size > 1
        ? decode_beam(input_tokens, audio_features, n_audio_frames, params, output_tokens)
        : decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens);
    if (!decoded) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    result.tokens = output_tokens;
    size_t text_start = 0;
    if (const language_prompt_entry * forced = find_language_prompt(params.language)) {
        // the prompt already contained "language <Name><asr_text>": output is text only
        result.language = forced->name;
    } else {
        // the model emits "language <Name><asr_text>" before the transcript; the
        // name can be more than one token (Cantonese, Macedonian), so skip up to
        // and including the <asr_text> token instead of a fixed count
        const size_t limit = std::min<size_t>(output_tokens.size(), 6);
        for (size_t i = 0; i < limit; ++i) {
            if (output_tokens[i] == ASR_TEXT_TOKEN_ID) {
                text_start = i + 1;
                break;
            }
        }
        if (text_start == 0) {
            text_start = std::min<size_t>(output_tokens.size(), 2);  // legacy fallback
        }
        if (text_start >= 2) {
            std::vector<int32_t> lang_tokens(output_tokens.begin() + 1, output_tokens.begin() + (text_start - 1));
            result.language = decoder_.decode_tokens(lang_tokens);
        }
    }
    std::vector text_tokens(output_tokens.begin() + text_start, output_tokens.end());
    result.text = decoder_.decode_tokens(text_tokens);
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        GGML_LOG_INFO("%s: [Timing]\n", __func__);
        GGML_LOG_INFO("    Mel spectrogram: %lld ms\n", static_cast<long long>(result.t_mel_ms));
        GGML_LOG_INFO("    Audio encoding:  %lld ms\n", static_cast<long long>(result.t_encode_ms));
        GGML_LOG_INFO("    Text decoding:   %lld ms\n", static_cast<long long>(result.t_decode_ms));
        GGML_LOG_INFO("    Total:           %lld ms\n", static_cast<long long>(result.t_total_ms));
        GGML_LOG_INFO("    Tokens generated: %zu\n", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 20);
    
    // Chat template format:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    
    // Token IDs from Qwen3 tokenizer:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // system = 8948
    // user = 872
    // assistant = 77091
    // \n = 198
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // <|audio_start|><|audio_pad|>...<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    // Forcing the language works like the qwen-asr Python package: the
    // assistant turn is pre-filled with "language <Name><asr_text>" so the
    // model only has to produce the transcript. Without it the model picks the
    // language itself, which is unreliable on short clips (e.g. Danish speech
    // transcribed as Swedish).
    if (const language_prompt_entry * forced = find_language_prompt(language)) {
        tokens.push_back(LANGUAGE_PROMPT_TOKEN_ID);
        tokens.insert(tokens.end(), forced->tokens.begin(), forced->tokens.end());
        tokens.push_back(ASR_TEXT_TOKEN_ID);
    }
    
    return tokens;
}


// Debug aid: QWEN3_ASR_DEBUG_TOPK=<n> prints the n highest logits of every
// decoding step to stderr so a run can be compared with the reference model.
static void debug_print_topk(const TextDecoder & decoder, const float * logits, int32_t vocab_size, size_t step) {
    static const int topk = [] {
        const char * env = getenv("QWEN3_ASR_DEBUG_TOPK");
        return env ? atoi(env) : 0;
    }();
    if (topk <= 0) {
        return;
    }
    std::vector<int32_t> idx(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) idx[i] = i;
    const int k = std::min<int32_t>(topk, vocab_size);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    fprintf(stderr, "step %zu:", step);
    for (int i = 0; i < k; ++i) {
        const std::string tok = decoder.decode_token(idx[i]);
        fprintf(stderr, " %d'%s'=%.2f", idx[i], tok.c_str(), logits[idx[i]]);
    }
    fprintf(stderr, "\n");
}


// Beam search over the text decoder. Every beam owns a KV cache; a beam that keeps a
// single child continues in its own cache, extra children copy the parent's cache
// (device side). Scores are summed token log-probabilities, finished hypotheses are
// compared by their per-token average so long and short outputs compete fairly.
namespace {

struct beam_hyp {
    std::vector<int32_t> tokens;
    double score = 0.0;   // sum of log-probabilities
    int32_t cache = -1;   // index into the cache pool
};

struct beam_candidate {
    int32_t parent;
    int32_t token;
    double score;
};

// top-k (token, logp) of a logit row; logp = logit - logsumexp
void topk_logprobs(const float * logits, int32_t vocab_size, int32_t k,
                   std::vector<std::pair<int32_t, double>> & out) {
    double max_logit = -INFINITY;
    for (int32_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, (double)logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        sum += exp((double)logits[i] - max_logit);
    }
    const double lse = max_logit + log(sum);
    std::vector<int32_t> idx(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) idx[i] = i;
    k = std::min(k, vocab_size);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    out.clear();
    for (int32_t i = 0; i < k; ++i) {
        out.emplace_back(idx[i], (double)logits[idx[i]] - lse);
    }
}

} // namespace

bool Qwen3ASR::decode_beam(const std::vector<int32_t> & input_tokens,
                           const std::vector<float> & audio_features,
                           int32_t n_audio_frames,
                           const transcribe_params & params,
                           std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    const int32_t B = std::max(1, params.beam_size);
    const int32_t vocab_size = cfg.vocab_size;
    const int32_t n_input = input_tokens.size();
    const int32_t n_ctx_needed = n_input + params.max_tokens;
    
    auto is_end = [&](int32_t tok) { return tok == cfg.eos_token_id || tok == cfg.pad_token_id; };
    
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    // prompt pass on the main cache
    std::vector<float> logits;
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(input_tokens.data(), n_input,
                                         audio_features.data(), n_audio_frames,
                                         /*audio_start_pos=*/9, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    // one cache per beam, all starting as copies of the prompt cache
    std::vector<kv_cache> pool(B);
    auto cleanup = [&]() { for (auto & c : pool) free_kv_cache(c); };
    for (int32_t i = 0; i < B; ++i) {
        if (!decoder_.alloc_kv_cache(pool[i], n_ctx_needed) ||
            !decoder_.copy_kv_cache(decoder_.get_kv_cache(), pool[i])) {
            error_msg_ = "Beam search: " + decoder_.get_error();
            cleanup();
            return false;
        }
    }
    
    std::vector<std::pair<int32_t, double>> topk;
    topk_logprobs(logits.data(), vocab_size, B, topk);
    
    std::vector<beam_hyp> beams;
    std::vector<std::pair<std::vector<int32_t>, double>> finished;  // tokens (no EOS), normalised score
    for (int32_t i = 0; i < (int32_t)topk.size(); ++i) {
        if (is_end(topk[i].first)) {
            finished.emplace_back(std::vector<int32_t>{}, topk[i].second);
            continue;
        }
        beam_hyp h;
        h.tokens.push_back(topk[i].first);
        h.score = topk[i].second;
        h.cache = i;
        beams.push_back(std::move(h));
    }
    
    std::vector<beam_candidate> candidates;
    while (!beams.empty() && (int32_t)finished.size() < B) {
        // every live beam feeds its last token through its own cache
        candidates.clear();
        bool max_reached = false;
        for (int32_t bi = 0; bi < (int32_t)beams.size(); ++bi) {
            beam_hyp & h = beams[bi];
            if ((int32_t)h.tokens.size() >= params.max_tokens) {
                max_reached = true;
                break;
            }
            const int32_t n_past = n_input + (int32_t)h.tokens.size() - 1;
            if (!decoder_.forward_with_cache(pool[h.cache], &h.tokens.back(), 1, n_past, logits)) {
                error_msg_ = "Beam search forward failed: " + decoder_.get_error();
                cleanup();
                return false;
            }
            topk_logprobs(logits.data(), vocab_size, B, topk);
            for (const auto & [tok, lp] : topk) {
                candidates.push_back({bi, tok, h.score + lp});
            }
        }
        if (max_reached) {
            break;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const beam_candidate & a, const beam_candidate & b) { return a.score > b.score; });
        
        // finished hypotheses leave the beam, the best B live ones continue
        std::vector<beam_candidate> next;
        for (const auto & c : candidates) {
            if ((int32_t)next.size() >= B) {
                break;
            }
            if (is_end(c.token)) {
                const auto & parent = beams[c.parent];
                finished.emplace_back(parent.tokens, c.score / (double)(parent.tokens.size() + 1));
                continue;
            }
            next.push_back(c);
        }
        
        // cache assignment: first child of a parent inherits its cache, further
        // children copy it; caches of parents without children are recycled
        std::vector<int32_t> children_seen(beams.size(), 0);
        std::vector<int32_t> free_caches;
        {
            std::vector<bool> used(B, false);
            for (const auto & c : next) {
                if (!children_seen[c.parent]++) {
                    used[beams[c.parent].cache] = true;
                }
            }
            for (int32_t i = 0; i < B; ++i) {
                if (!used[i]) free_caches.push_back(i);
            }
        }
        std::fill(children_seen.begin(), children_seen.end(), 0);
        std::vector<beam_hyp> new_beams;
        new_beams.reserve(next.size());
        for (const auto & c : next) {
            const beam_hyp & parent = beams[c.parent];
            beam_hyp h;
            h.tokens = parent.tokens;
            h.tokens.push_back(c.token);
            h.score = c.score;
            if (children_seen[c.parent]++ == 0) {
                h.cache = parent.cache;
            } else {
                h.cache = free_caches.back();
                free_caches.pop_back();
                if (!decoder_.copy_kv_cache(pool[parent.cache], pool[h.cache])) {
                    error_msg_ = "Beam search cache copy failed: " + decoder_.get_error();
                    cleanup();
                    return false;
                }
            }
            new_beams.push_back(std::move(h));
        }
        beams.swap(new_beams);
        
        if (progress_callback_ && !beams.empty()) {
            progress_callback_(beams.front().tokens.size(), params.max_tokens);
        }
    }
    
    // best finished hypothesis by per-token score, else best live beam
    const std::vector<int32_t> * best = nullptr;
    double best_score = -INFINITY;
    for (const auto & f : finished) {
        if (f.second > best_score) {
            best_score = f.second;
            best = &f.first;
        }
    }
    if (!best) {
        for (const auto & h : beams) {
            const double norm = h.score / (double)std::max<size_t>(1, h.tokens.size());
            if (norm > best_score) {
                best_score = norm;
                best = &h.tokens;
            }
        }
    }
    output_tokens = best ? *best : std::vector<int32_t>{};
    cleanup();
    return true;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                              const std::vector<float> & audio_features,
                              int32_t n_audio_frames,
                              const transcribe_params & params,
                              std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    // That's 8 tokens before the first audio_pad
    int32_t audio_start_pos = 9;
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    // Debug aid: QWEN3_ASR_DEBUG_LOGITS=<file> dumps the prompt logits of every
    // position (n_tokens x vocab_size, float32, row-major) for comparison with
    // the reference model.
    if (const char * dump_path = getenv("QWEN3_ASR_DEBUG_LOGITS")) {
        std::ofstream dump(dump_path, std::ios::binary);
        dump.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
        fprintf(stderr, "dumped %zu logits (%zu rows) to %s\n", logits.size(), logits.size() / vocab_size, dump_path);
    }
    debug_print_topk(decoder_, logits.data(), vocab_size, 0);
    int32_t next_token = sample_greedy(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_tokens.push_back(next_token);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        debug_print_topk(decoder_, logits.data(), vocab_size, output_tokens.size());
        next_token = sample_greedy(logits.data(), vocab_size);
        output_tokens.push_back(next_token);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            GGML_LOG_INFO("Generated %zu tokens...\n", output_tokens.size());
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
    }
    
    return true;
}

int32_t Qwen3ASR::sample_greedy(const float * logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    return load_wav(path, samples, sample_rate);
}

} // namespace qwen3_asr
