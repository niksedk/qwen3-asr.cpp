#pragma once
// Generated from the Qwen3-ASR tokenizer: token ids of " <Language>" for every
// supported language, used to force the output language in the prompt
// ("language <Language><asr_text>"), exactly like the qwen-asr Python package.
#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_asr {

struct language_prompt_entry {
    const char * name;              // canonical name as the model emits it
    std::vector<const char *> aliases;  // lowercase names / ISO codes accepted on input
    std::vector<int32_t> tokens;    // token ids of " <Name>"
};

static const int32_t LANGUAGE_PROMPT_TOKEN_ID = 11528;   // "language"
static const int32_t ASR_TEXT_TOKEN_ID = 151704;         // "<asr_text>"

static const language_prompt_entry LANGUAGE_PROMPT_TABLE[] = {
    {"Chinese", {"chinese", "zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "cmn", "mandarin"}, {8453}},
    {"English", {"english", "en"}, {6364}},
    {"Cantonese", {"cantonese", "yue"}, {72366, 2367}},
    {"Arabic", {"arabic", "ar"}, {34117}},
    {"German", {"german", "de"}, {5938}},
    {"French", {"french", "fr"}, {8585}},
    {"Spanish", {"spanish", "es"}, {15154}},
    {"Portuguese", {"portuguese", "pt"}, {42188}},
    {"Indonesian", {"indonesian", "id"}, {58829}},
    {"Italian", {"italian", "it"}, {14811}},
    {"Korean", {"korean", "ko", "kr"}, {16134}},
    {"Russian", {"russian", "ru"}, {8522}},
    {"Thai", {"thai", "th"}, {26392}},
    {"Vietnamese", {"vietnamese", "vi"}, {48477}},
    {"Japanese", {"japanese", "ja", "jp"}, {10769}},
    {"Turkish", {"turkish", "tr"}, {23734}},
    {"Hindi", {"hindi", "hi"}, {43980}},
    {"Malay", {"malay", "ms"}, {79140}},
    {"Dutch", {"dutch", "nl"}, {23234}},
    {"Swedish", {"swedish", "sv"}, {30109}},
    {"Danish", {"danish", "da"}, {43680}},
    {"Finnish", {"finnish", "fi"}, {57853}},
    {"Polish", {"polish", "pl"}, {31984}},
    {"Czech", {"czech", "cs"}, {33150}},
    {"Filipino", {"filipino", "fil", "tl", "tagalog"}, {62417}},
    {"Persian", {"persian", "fa", "farsi"}, {49861}},
    {"Greek", {"greek", "el"}, {17860}},
    {"Romanian", {"romanian", "ro"}, {73597}},
    {"Hungarian", {"hungarian", "hu"}, {56769}},
    {"Macedonian", {"macedonian", "mk"}, {56452, 75491}},
};

} // namespace qwen3_asr
