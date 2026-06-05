// server.js - OpenAI Multi-Backend API Proxy (NVIDIA NIM + AgentRouter)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase payload limit
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// 🔧 Normalize URL path to lowercase (fixes /V1/ vs /v1/ case sensitivity issues)
app.use((req, res, next) => {
    req.url = req.url.toLowerCase();
    next();
});

// =============================================
// 🔧 BACKEND CONFIGURATION
// =============================================

// Backend 1: NVIDIA NIM
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Backend 2: AgentRouter (OpenAI-compatible, free $175 credits)
const AGENTROUTER_API_BASE = process.env.AGENTROUTER_API_BASE || 'https://agentrouter.org/v1';
const AGENTROUTER_API_KEY = process.env.AGENTROUTER_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 Estimate token count from text (~4 chars per token for English, ~3 for mixed)
function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.length / 3.5);
}

// 🔧 Fix paragraph formatting for newer models (DeepSeek v4, GLM 5.1+)
// Some models return literal '\n' escape sequences instead of real newlines,
// or collapse all paragraphs into a single block of text.
function fixParagraphs(text) {
    if (!text) return text;
    // Replace literal \n (escaped) with real newlines
    let fixed = text.replace(/\\n/g, '\n');
    // Replace literal \r\n or \r with real newlines
    fixed = fixed.replace(/\\r\\n/g, '\n').replace(/\\r/g, '\n');
    return fixed;
}

// =============================================
// 🔧 MODEL MAPPING — Dual Backend Routing
// =============================================
// Each entry maps an OpenAI model name to:
//   - backend: 'nvidia' or 'agentrouter'
//   - model: the actual model ID to send to that backend

const MODEL_MAPPING = {
<<<<<<< HEAD
    // === NVIDIA NIM Backend — DeepSeek, Llama, Mistral ===
    'gpt-4-turbo':         { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-pro' },      // 🔥 1M context, best DeepSeek
    'gpt-4-1106-preview':  { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-pro' },
    'gpt-4':               { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-flash' },     // 284B MoE, 1M context, fast
    'gpt-4-turbo-preview': { backend: 'nvidia', model: 'deepseek-ai/deepseek-v3.2' },         // 685B, still working
    'gpt-4-32k':           { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },
    'gpt-3.5-turbo':       { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },
    'gpt-3.5-turbo-16k':   { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },

    // === AgentRouter Backend — GLM, Claude ===
    'gpt-4o':              { backend: 'agentrouter', model: 'glm-5.1' },                      // 🔥 GLM 5.1 via AgentRouter (was broken on NVIDIA)
    'gpt-4o-mini':         { backend: 'agentrouter', model: 'claude-sonnet-4-6' },             // 🔥 Claude Sonnet 4.6 via AgentRouter
=======
    // === 2026 UPDATED MAPPINGS — May 14 ===
    // Tier 1: Best quality (use for main RP)
    'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro',     // 🔥 NEW — 1M context, best DeepSeek
    'gpt-4-1106-preview': 'google/gemma-4-31b-it',

    // Tier 2: Fast & smart
    'gpt-4o': 'moonshotai/kimi-k2.6',                          // 🔥 Replaces deprecated GLM 4.7
    'gpt-4': 'google/gemma-4-31b-it',           // 284B MoE, 1M context, fast
    'gpt-4-turbo-preview': 'deepseek-ai/deepseek-v3.2', // 685B, still working

    // Tier 3: Lighter / fallback
    'gpt-4o-mini': 'mistralai/mistral-medium-3.5-128b',  // 128B, solid for chat
    'gpt-4-32k': 'meta/llama-3.3-70b-instruct',          // Llama, still working
    'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
    'gpt-3.5-turbo-16k': 'meta/llama-3.3-70b-instruct',
>>>>>>> 91eafa71f6c5b69a41577591fe5b16d78cf6a24b
};

// 🔥 Context window sizes per model (in tokens)
const MODEL_CONTEXT_SIZES = {
<<<<<<< HEAD
    // NVIDIA models
    'deepseek-ai/deepseek-v4-pro': 1000000,
    'deepseek-ai/deepseek-v4-flash': 1000000,
=======
    // New 2026 models
    'deepseek-ai/deepseek-v4-pro': 1000000,     // 1M tokens! 🔥
    'deepseek-ai/deepseek-v4-flash': 1000000,   // 1M tokens
    'moonshotai/kimi-k2.6': 131072,                      // ~131K (estimated, same family as GLM 4.7)
    'moonshotai/kimi-k2.6': 128000,              // estimated
    'mistralai/mistral-medium-3.5-128b': 128000,
    'minimaxai/minimax-m2.7': 128000,
    // Older models still active
>>>>>>> 91eafa71f6c5b69a41577591fe5b16d78cf6a24b
    'deepseek-ai/deepseek-v3.2': 128000,
    'deepseek-ai/deepseek-r1': 164000,
    'meta/llama-3.3-70b-instruct': 128000,
    'meta/llama-3.1-70b-instruct': 128000,
    'meta/llama-3.1-405b-instruct': 128000,
    'mistralai/mistral-medium-3.5-128b': 128000,
    // AgentRouter models
    'glm-5.1': 131072,
    'claude-sonnet-4-5': 200000,
    'claude-sonnet-4-6': 200000,
};
const DEFAULT_CONTEXT_SIZE = 128000;
const RESPONSE_RESERVE_TOKENS = 4096;

// =============================================
// 🔧 BACKEND ROUTING HELPER
// =============================================
function getBackendConfig(backendName) {
    if (backendName === 'agentrouter') {
        return {
            baseUrl: AGENTROUTER_API_BASE,
            apiKey: AGENTROUTER_API_KEY,
            name: 'AgentRouter',
        };
    }
    // Default: NVIDIA NIM
    return {
        baseUrl: NIM_API_BASE,
        apiKey: NIM_API_KEY,
        name: 'NVIDIA NIM',
    };
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI Multi-Backend Proxy (NVIDIA NIM + AgentRouter)',
        updated: '2026-06-05',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE,
        backends: {
            nvidia: { configured: !!NIM_API_KEY, base: NIM_API_BASE },
            agentrouter: { configured: !!AGENTROUTER_API_KEY, base: AGENTROUTER_API_BASE },
        },
        top_models: {
<<<<<<< HEAD
            'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro → NVIDIA (1M ctx)',
            'gpt-4o':      'glm-5.1 → AgentRouter (131K ctx)',
            'gpt-4':       'deepseek-ai/deepseek-v4-flash → NVIDIA (1M ctx)',
            'gpt-4o-mini': 'claude-sonnet-4-6 → AgentRouter (200K ctx)',
=======
            'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro (1M ctx)',
            'gpt-4o': 'moonshotai/kimi-k2.6 (131K ctx)',
            'gpt-4': 'google/gemma-4-31b-it (1M ctx)',
            'gpt-4o-mini': 'mistralai/mistral-medium-3.5-128b (128K ctx)',
>>>>>>> 91eafa71f6c5b69a41577591fe5b16d78cf6a24b
        }
    });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(model => ({
        id: model,
        object: 'model',
        created: Date.now(),
        owned_by: 'multi-backend-proxy'
    }));

    res.json({
        object: 'list',
        data: models
    });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
    try {
        const { model, messages, temperature, max_tokens, stream } = req.body;

        // Validate request
        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({
                error: {
                    message: 'Messages array is required and must not be empty',
                    type: 'invalid_request_error',
                    code: 400
                }
            });
        }

        // Smart model selection with fallback
        let mapping = MODEL_MAPPING[model];
        if (!mapping) {
            const modelLower = (model || '').toLowerCase();
            if (modelLower.includes('gpt-4') || modelLower.includes('claude-3') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
                mapping = { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-pro' };
            } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
                mapping = { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-flash' };
            } else {
                mapping = { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' };
            }
            console.log(`⚠️ Unknown model "${model}" → fallback to ${mapping.model} (${mapping.backend})`);
        }

        const targetModel = mapping.model;
        const backend = getBackendConfig(mapping.backend);

        // Check that the backend API key is configured
        if (!backend.apiKey) {
            console.error(`❌ Backend ${backend.name} has no API key configured!`);
            return res.status(500).json({
                error: {
                    message: `Backend ${backend.name} is not configured (missing API key). Set ${mapping.backend === 'agentrouter' ? 'AGENTROUTER_API_KEY' : 'NIM_API_KEY'} environment variable.`,
                    type: 'configuration_error',
                    code: 500
                }
            });
        }

        // 🔥 TOKEN-AWARE CONTEXT MANAGEMENT
        const contextSize = MODEL_CONTEXT_SIZES[targetModel] || DEFAULT_CONTEXT_SIZE;
        const maxInputTokens = contextSize - RESPONSE_RESERVE_TOKENS;

        // 1. Always keep ALL system messages
        const systemMessages = messages.filter(msg => msg.role === 'system');
        const nonSystemMessages = messages.filter(msg => msg.role !== 'system');

        // 2. Calculate tokens used by system messages
        let usedTokens = 0;
        const cleanedSystemMessages = systemMessages.map(msg => {
            const cleaned = { role: msg.role, content: msg.content || '' };
            usedTokens += estimateTokens(cleaned.content) + 4;
            return cleaned;
        });

        // 3. Fill remaining context with recent conversation messages (newest first)
        const remainingTokenBudget = maxInputTokens - usedTokens;
        const cleanedConversation = [];
        let conversationTokens = 0;

        for (let i = nonSystemMessages.length - 1; i >= 0; i--) {
            const msg = nonSystemMessages[i];
            const cleaned = { role: msg.role, content: msg.content || '' };
            const msgTokens = estimateTokens(cleaned.content) + 4;

            if (conversationTokens + msgTokens > remainingTokenBudget) {
                break;
            }
            cleanedConversation.unshift(cleaned);
            conversationTokens += msgTokens;
        }

        // 4. Combine: system messages first, then conversation
        const cleanedMessages = [...cleanedSystemMessages, ...cleanedConversation];

        console.log(`📊 Context: ${usedTokens + conversationTokens}/${maxInputTokens} tokens | ${cleanedSystemMessages.length} system + ${cleanedConversation.length}/${nonSystemMessages.length} conversation msgs`);

        // Build request (OpenAI format — works for both backends)
        const apiRequest = {
            model: targetModel,
            messages: cleanedMessages,
            temperature: temperature || 0.6,
            max_tokens: Math.min(max_tokens || 2048, 8192),
            stream: stream || false
        };

        // Only add thinking mode for NVIDIA backend
        if (ENABLE_THINKING_MODE && mapping.backend === 'nvidia') {
            apiRequest.extra_body = { chat_template_kwargs: { thinking: true } };
        }

        // 🔍 DEBUG logging
        const payloadJson = JSON.stringify(apiRequest);
        const payloadSizeKB = (Buffer.byteLength(payloadJson, 'utf8') / 1024).toFixed(1);
        console.log(`🔍 DEBUG: Payload ${payloadSizeKB} KB | Model: ${targetModel} | Backend: ${backend.name} | Messages: ${cleanedMessages.length} | Stream: ${stream || false}`);

        // Make request to the selected backend
        const response = await axios.post(`${backend.baseUrl}/chat/completions`, apiRequest, {
            headers: {
                'Authorization': `Bearer ${backend.apiKey}`,
                'Content-Type': 'application/json'
            },
            responseType: stream ? 'stream' : 'json',
            timeout: 120000
        });

        if (stream) {
            // Handle streaming response
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            let buffer = '';
            let reasoningStarted = false;

            response.data.on('data', (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        if (line.includes('[DONE]')) {
                            res.write(line + '\n');
                            return;
                        }

                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.choices?.[0]?.delta) {
                                const reasoning = data.choices[0].delta.reasoning_content;
                                const content = data.choices[0].delta.content;

                                if (SHOW_REASONING) {
                                    let combinedContent = '';

                                    if (reasoning && !reasoningStarted) {
                                        combinedContent = '<think>\n' + reasoning;
                                        reasoningStarted = true;
                                    } else if (reasoning) {
                                        combinedContent = reasoning;
                                    }

                                    if (content && reasoningStarted) {
                                        combinedContent += '</think>\n\n' + content;
                                        reasoningStarted = false;
                                    } else if (content) {
                                        combinedContent += content;
                                    }

                                    if (combinedContent) {
                                        data.choices[0].delta.content = combinedContent;
                                        delete data.choices[0].delta.reasoning_content;
                                    }
                                } else {
                                    if (content) {
                                        data.choices[0].delta.content = fixParagraphs(content);
                                    } else {
                                        data.choices[0].delta.content = '';
                                    }
                                    delete data.choices[0].delta.reasoning_content;
                                }
                            }
                            res.write(`data: ${JSON.stringify(data)}\n\n`);
                        } catch (e) {
                            res.write(line + '\n');
                        }
                    }
                });
            });

            response.data.on('end', () => res.end());
            response.data.on('error', (err) => {
                console.error('Stream error:', err);
                res.end();
            });
        } else {
            // Transform response to OpenAI format
            const openaiResponse = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: response.data.choices.map(choice => {
                    let fullContent = fixParagraphs(choice.message?.content || '');

                    if (SHOW_REASONING && choice.message?.reasoning_content) {
                        fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
                    }

                    return {
                        index: choice.index,
                        message: {
                            role: choice.message.role,
                            content: fullContent
                        },
                        finish_reason: choice.finish_reason
                    };
                }),
                usage: response.data.usage || {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0
                }
            };

            res.json(openaiResponse);
        }

    } catch (error) {
        // 🔍 Safe error logging
        const status = error.response?.status || 'N/A';
        const isTimeout = error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT';

        let errorDetail = error.message || 'unknown error';
        try {
            const backendError = error.response?.data;
            if (backendError && typeof backendError === 'string') {
                errorDetail = backendError.slice(0, 500);
            } else if (backendError && typeof backendError === 'object') {
                const extracted = backendError.detail || backendError.error?.message || backendError.message;
                if (typeof extracted === 'string') {
                    errorDetail = extracted;
                }
            }
        } catch (e) {
            // Keep the default error.message
        }

        if (isTimeout) {
            errorDetail = `Backend API timeout after 120s — the model may be overloaded. Original: ${error.message}`;
        }

        console.error(`❌ Proxy error: status=${status} | timeout=${isTimeout} | ${errorDetail}`);

        res.status(error.response?.status || (isTimeout ? 504 : 500)).json({
            error: {
                message: errorDetail,
                type: isTimeout ? 'timeout_error' : 'invalid_request_error',
                code: error.response?.status || (isTimeout ? 504 : 500)
            }
        });
    }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
    res.status(404).json({
        error: {
            message: `Endpoint ${req.path} not found`,
            type: 'invalid_request_error',
            code: 404
        }
    });
});

app.listen(PORT, () => {
    console.log(`OpenAI Multi-Backend Proxy running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Backends: NVIDIA NIM ${NIM_API_KEY ? '✅' : '❌'} | AgentRouter ${AGENTROUTER_API_KEY ? '✅' : '❌'}`);
    console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
    console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
