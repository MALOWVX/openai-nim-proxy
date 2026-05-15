// skerver.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase payload limit
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

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

// Model mapping (adjust based on available NIM models)
// NOTE: Some models like deepseek-v3_1-terminus have 404 issues on NVIDIA API
// Using confirmed working models instead
const MODEL_MAPPING = {
    // === 2026 UPDATED MAPPINGS — May 14 ===
    // Tier 1: Best quality (use for main RP)
    'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro',     // 🔥 NEW — 1M context, best DeepSeek
    'gpt-4-1106-preview': 'deepseek-ai/deepseek-v4-pro',

    // Tier 2: Fast & smart
    'gpt-4o': 'z-ai/glm-5.1',                          // 🔥 Replaces deprecated GLM 4.7
    'gpt-4': 'deepseek-ai/deepseek-v4-flash',           // 284B MoE, 1M context, fast
    'gpt-4-turbo-preview': 'deepseek-ai/deepseek-v3.2', // 685B, still working

    // Tier 3: Lighter / fallback
    'gpt-4o-mini': 'mistralai/mistral-medium-3.5-128b',  // 128B, solid for chat
    'gpt-4-32k': 'meta/llama-3.3-70b-instruct',          // Llama, still working
    'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
    'gpt-3.5-turbo-16k': 'meta/llama-3.3-70b-instruct',
};

// 🔥 Context window sizes per NIM model (in tokens)
// Set to the REAL model max — Janitor.ai already limits on its side,
// this acts as a safety net if the client sends too much
const MODEL_CONTEXT_SIZES = {
    // New 2026 models
    'deepseek-ai/deepseek-v4-pro': 1000000,     // 1M tokens! 🔥
    'deepseek-ai/deepseek-v4-flash': 1000000,   // 1M tokens
    'z-ai/glm-5.1': 131072,                      // ~131K (estimated, same family as GLM 4.7)
    'moonshotai/kimi-k2.6': 128000,              // estimated
    'mistralai/mistral-medium-3.5-128b': 128000,
    'minimaxai/minimax-m2.7': 128000,
    // Older models still active
    'deepseek-ai/deepseek-v3.2': 128000,
    'deepseek-ai/deepseek-r1': 164000,
    'meta/llama-3.3-70b-instruct': 128000,
    'meta/llama-3.1-70b-instruct': 128000,
    'meta/llama-3.1-405b-instruct': 128000,
};
const DEFAULT_CONTEXT_SIZE = 128000;  // Most models support at least 128K now
const RESPONSE_RESERVE_TOKENS = 4096; // Reserve for model output

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        updated: '2026-05-14',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE,
        top_models: {
            'gpt-4-turbo': 'deepseek-ai/deepseek-v4-pro (1M ctx)',
            'gpt-4o':      'z-ai/glm-5.1 (131K ctx)',
            'gpt-4':       'deepseek-ai/deepseek-v4-flash (1M ctx)',
            'gpt-4o-mini': 'mistralai/mistral-medium-3.5-128b (128K ctx)',
        }
    });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(model => ({
        id: model,
        object: 'model',
        created: Date.now(),
        owned_by: 'nvidia-nim-proxy'
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

        // Smart model selection with fallback (moved BEFORE message trimming to know context size)
        let nimModel = MODEL_MAPPING[model];
        if (!nimModel) {
            // No probe test — just use smart fallback based on model name
            const modelLower = (model || '').toLowerCase();
            if (modelLower.includes('gpt-4') || modelLower.includes('claude-3') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
                nimModel = 'deepseek-ai/deepseek-v4-pro';    // Best: 1M context
            } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
                nimModel = 'deepseek-ai/deepseek-v4-flash';  // Fast: 284B MoE, 1M context
            } else {
                nimModel = 'meta/llama-3.3-70b-instruct';    // Safe default
            }
            console.log(`⚠️ Unknown model "${model}" → fallback to ${nimModel}`);
        }

        // 🔥 TOKEN-AWARE CONTEXT MANAGEMENT
        // Use the model's actual context window instead of an arbitrary message limit
        const contextSize = MODEL_CONTEXT_SIZES[nimModel] || DEFAULT_CONTEXT_SIZE;
        const maxInputTokens = contextSize - RESPONSE_RESERVE_TOKENS;

        // 1. Always keep ALL system messages (bot identity, hero descriptions, etc.)
        const systemMessages = messages.filter(msg => msg.role === 'system');
        const nonSystemMessages = messages.filter(msg => msg.role !== 'system');

        // 2. Calculate tokens used by system messages
        let usedTokens = 0;
        const cleanedSystemMessages = systemMessages.map(msg => {
            const cleaned = { role: msg.role, content: msg.content || '' };
            usedTokens += estimateTokens(cleaned.content) + 4; // +4 for role/formatting overhead
            return cleaned;
        });

        // 3. Fill remaining context with recent conversation messages (newest first)
        const remainingTokenBudget = maxInputTokens - usedTokens;
        const cleanedConversation = [];
        let conversationTokens = 0;

        // Iterate from newest to oldest, keep as many as fit
        for (let i = nonSystemMessages.length - 1; i >= 0; i--) {
            const msg = nonSystemMessages[i];
            const cleaned = { role: msg.role, content: msg.content || '' };
            const msgTokens = estimateTokens(cleaned.content) + 4;

            if (conversationTokens + msgTokens > remainingTokenBudget) {
                break; // No more room
            }
            cleanedConversation.unshift(cleaned); // Add to front to maintain order
            conversationTokens += msgTokens;
        }

        // 4. Combine: system messages first, then conversation
        const cleanedMessages = [...cleanedSystemMessages, ...cleanedConversation];

        console.log(`📊 Context: ${usedTokens + conversationTokens}/${maxInputTokens} tokens | ${cleanedSystemMessages.length} system + ${cleanedConversation.length}/${nonSystemMessages.length} conversation msgs`);


        // Transform OpenAI request to NIM format
        const nimRequest = {
            model: nimModel,
            messages: cleanedMessages, // Use cleaned messages
            temperature: temperature || 0.6,
            max_tokens: Math.min(max_tokens || 2048, 8192), // Limit max_tokens
            stream: stream || false
        };

        // Only add extra_body if thinking mode is enabled
        if (ENABLE_THINKING_MODE) {
            nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
        }

        // 🔍 DEBUG: Log payload size to diagnose large request issues
        const payloadJson = JSON.stringify(nimRequest);
        const payloadSizeKB = (Buffer.byteLength(payloadJson, 'utf8') / 1024).toFixed(1);
        console.log(`🔍 DEBUG: Payload size: ${payloadSizeKB} KB | Model: ${nimModel} | Messages: ${cleanedMessages.length} | Stream: ${stream || false}`);

        console.log('Sending to NVIDIA:', { model: nimModel, messageCount: cleanedMessages.length });

        // Make request to NVIDIA NIM API (120s timeout to avoid infinite hangs)
        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: {
                'Authorization': `Bearer ${NIM_API_KEY}`,
                'Content-Type': 'application/json'
            },
            responseType: stream ? 'stream' : 'json',
            timeout: 120000 // 120 seconds
        });

        if (stream) {
            // Handle streaming response with reasoning
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
            // Transform NIM response to OpenAI format with reasoning
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
        // 🔍 Safe error logging — handles circular references, streams, HTML responses
        const status = error.response?.status || 'N/A';
        const isTimeout = error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT';

        // Safely extract error detail from various response shapes
        let errorDetail = error.message || 'unknown error';
        try {
            const nvidiaError = error.response?.data;
            if (nvidiaError && typeof nvidiaError === 'string') {
                errorDetail = nvidiaError.slice(0, 500);
            } else if (nvidiaError && typeof nvidiaError === 'object') {
                // Only access simple string properties — avoid anything that could be a stream/socket
                const extracted = nvidiaError.detail || nvidiaError.error?.message || nvidiaError.message;
                if (typeof extracted === 'string') {
                    errorDetail = extracted;
                }
            }
        } catch (e) {
            // Keep the default error.message
        }

        if (isTimeout) {
            errorDetail = `NVIDIA API timeout after 120s — the model may be overloaded. Original: ${error.message}`;
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
    console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
    console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});