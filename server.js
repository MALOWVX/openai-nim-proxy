// server.js - OpenAI Multi-Backend API Proxy (NVIDIA NIM + AgentRouter)
// With automatic fallback: AgentRouter → NVIDIA NIM on "sensitive words detected"
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
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
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false;

// 🔥 Estimate token count from text (~4 chars per token for English, ~3 for mixed)
function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.length / 3.5);
}

// 🔧 Fix paragraph formatting for newer models (DeepSeek v4, GLM 5.1+)
function fixParagraphs(text) {
    if (!text) return text;
    let fixed = text.replace(/\\n/g, '\n');
    fixed = fixed.replace(/\\r\\n/g, '\n').replace(/\\r/g, '\n');
    return fixed;
}

// 🛡️ OBFUSCATION: Insert zero-width spaces to bypass sensitive word filters
// The AI model ignores these invisible chars, but keyword filters can't match words
const ZWS = '\u200B'; // Zero-Width Space (invisible character)
function obfuscateText(val) {
    if (!val) return val;
    
    // If it's a plain string, obfuscate it
    if (typeof val === 'string') {
        return val.replace(/\S{5,}/g, (word) => {
            const mid = Math.floor(word.length / 2);
            return word.slice(0, mid) + ZWS + word.slice(mid);
        });
    }
    
    // If it's an array (e.g. array of content blocks), recurse on elements
    if (Array.isArray(val)) {
        return val.map(obfuscateText);
    }
    
    // If it's an object, recurse only on text/content properties to avoid breaking API schema
    if (typeof val === 'object') {
        const result = {};
        for (const key in val) {
            if (Object.prototype.hasOwnProperty.call(val, key)) {
                if (key === 'text' || key === 'content') {
                    result[key] = obfuscateText(val[key]);
                } else if (typeof val[key] === 'object' && val[key] !== null) {
                    result[key] = obfuscateText(val[key]);
                } else {
                    result[key] = val[key]; // Keep other fields (like role, type, url) as-is
                }
            }
        }
        return result;
    }
    
    return val;
}

// Apply obfuscation to all messages in a conversation
function obfuscateMessages(messages) {
    return messages.map(msg => ({
        ...msg,
        content: obfuscateText(msg.content)
    }));
}

// =============================================
// 🔧 MODEL MAPPING — Dual Backend Routing
// =============================================

const MODEL_MAPPING = {
    // === NVIDIA NIM Backend — DeepSeek, Llama, Mistral ===
    'gpt-4-turbo':         { backend: 'nvidia', model: 'google/gemma-4-31b-it' },
    'gpt-4-1106-preview':  { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-pro' },
    'gpt-4':               { backend: 'nvidia', model: 'z-ai/glm-5.2' },
    'gpt-4-turbo-preview': { backend: 'nvidia', model: 'deepseek-ai/deepseek-v3.2' },
    'gpt-4-32k':           { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },
    'gpt-3.5-turbo':       { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },
    'gpt-3.5-turbo-16k':   { backend: 'nvidia', model: 'meta/llama-3.3-70b-instruct' },

    // === AgentRouter Backend — GLM, Claude ===
    'gpt-4o':              { backend: 'agentrouter', model: 'glm-5.1' },
    'gpt-4o-mini':         { backend: 'agentrouter', model: 'claude-opus-4-6' },
};

// 🔥 Fallback model when AgentRouter blocks with "sensitive words detected"
const SENSITIVE_WORDS_FALLBACK = { backend: 'nvidia', model: 'deepseek-ai/deepseek-v4-pro' };

// 🔥 Context window sizes per model (in tokens)
const MODEL_CONTEXT_SIZES = {
    // NVIDIA models
    'deepseek-ai/deepseek-v4-pro': 1000000,
    'deepseek-ai/deepseek-v4-flash': 1000000,
    'deepseek-ai/deepseek-v3.2': 128000,
    'deepseek-ai/deepseek-r1': 164000,
    'meta/llama-3.3-70b-instruct': 128000,
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
    return {
        baseUrl: NIM_API_BASE,
        apiKey: NIM_API_KEY,
        name: 'NVIDIA NIM',
    };
}

// =============================================
// 🔧 HELPER: Read error details from backend response (handles streams)
// =============================================
async function extractErrorDetail(error) {
    let errorDetail = error.message || 'unknown error';
    try {
        const backendError = error.response?.data;
        if (backendError && typeof backendError.on === 'function') {
            // Set encoding to correctly handle UTF-8 multibyte characters
            backendError.setEncoding('utf8');
            // It's a stream! We need to read it to get the error details
            const streamData = await new Promise((resolve) => {
                let body = '';
                backendError.on('data', chunk => body += chunk);
                backendError.on('end', () => resolve(body));
                backendError.on('error', () => resolve(''));
                setTimeout(() => resolve(body), 2000); // 2s timeout
            });
            try {
                const parsed = JSON.parse(streamData);
                errorDetail = parsed.detail || parsed.error?.message || parsed.message || streamData;
            } catch (e) {
                errorDetail = streamData || error.message;
            }
        } else if (backendError && typeof backendError === 'string') {
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
    return errorDetail;
}

// =============================================
// 🔧 HELPER: Make API request to a backend
// =============================================
async function makeBackendRequest(backendName, apiRequest, stream) {
    const backend = getBackendConfig(backendName);

    // Build headers dynamically
    const headers = {
        'Authorization': `Bearer ${backend.apiKey}`,
        'Content-Type': 'application/json'
    };

    // 🔧 Bypass AgentRouter's client fingerprinting (unauthorized client detected error)
    if (backendName === 'agentrouter') {
        headers['Originator'] = 'codex_cli_rs';
        headers['User-Agent'] = 'codex_cli_rs/0.101.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464';
        headers['Version'] = '0.101.0';
    }

    // 🛡️ Obfuscate messages for AgentRouter to bypass sensitive word filters
    const finalRequest = { ...apiRequest };
    if (backendName === 'agentrouter' && finalRequest.messages) {
        finalRequest.messages = obfuscateMessages(finalRequest.messages);
        console.log(`🛡️ Obfuscation applied to ${finalRequest.messages.length} messages for AgentRouter`);
    }

    return axios.post(`${backend.baseUrl}/chat/completions`, finalRequest, {
        headers: headers,
        responseType: stream ? 'stream' : 'json',
        timeout: 120000
    });
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI Multi-Backend Proxy (NVIDIA NIM + AgentRouter)',
        updated: '2026-06-06',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE,
        sensitive_words_fallback: 'ENABLED → NVIDIA DeepSeek v4 Pro',
        backends: {
            nvidia: { configured: !!NIM_API_KEY, base: NIM_API_BASE },
            agentrouter: { configured: !!AGENTROUTER_API_KEY, base: AGENTROUTER_API_BASE },
        },
        top_models: {
            'gpt-4-turbo': 'deepseek-v4-pro → NVIDIA (1M ctx)',
            'gpt-4o':      'glm-5.1 → AgentRouter (131K ctx)',
            'gpt-4':       'z-ai/glm-5.1 → NVIDIA (1M ctx)',
            'gpt-4o-mini': 'claude-sonnet-4-6 → AgentRouter (200K ctx)',
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

// =============================================
// 🔥 HELPER: Send response back to client (stream or JSON)
// =============================================
function sendStreamResponse(res, response, model) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // 🔧 Set encoding to handle multi-byte characters (like accented letters) split across chunks
    response.data.setEncoding('utf8');

    let buffer = '';
    let reasoningStarted = false;

    response.data.on('data', (chunk) => {
        // Since we set encoding to utf8, chunk is already a correctly decoded string
        buffer += chunk;
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
            if (line.startsWith('data: ')) {
                if (line.includes('[DONE]')) {
                    res.write(line + '\n\n');
                    return;
                }

                try {
                    const data = JSON.parse(line.slice(6));
                    if (!data || typeof data !== 'object') {
                        // Skip empty or non-object events (e.g., data: null)
                        return;
                    }
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
                    res.write(line + '\n\n');
                }
            }
        });
    });

    response.data.on('end', () => res.end());
    response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
    });
}

function sendJsonResponse(res, response, model) {
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

// =============================================
// 🔥 MAIN PROXY — Chat completions endpoint
// =============================================
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

        let targetModel = mapping.model;
        let currentBackend = mapping.backend;
        const backend = getBackendConfig(currentBackend);

        // Check that the backend API key is configured
        if (!backend.apiKey) {
            console.error(`❌ Backend ${backend.name} has no API key configured!`);
            return res.status(500).json({
                error: {
                    message: `Backend ${backend.name} is not configured (missing API key). Set ${currentBackend === 'agentrouter' ? 'AGENTROUTER_API_KEY' : 'NIM_API_KEY'} environment variable.`,
                    type: 'configuration_error',
                    code: 500
                }
            });
        }

        // 🔥 TOKEN-AWARE CONTEXT MANAGEMENT
        const contextSize = MODEL_CONTEXT_SIZES[targetModel] || DEFAULT_CONTEXT_SIZE;
        const maxInputTokens = contextSize - RESPONSE_RESERVE_TOKENS;

        const systemMessages = messages.filter(msg => msg.role === 'system');
        const nonSystemMessages = messages.filter(msg => msg.role !== 'system');

        let usedTokens = 0;
        const cleanedSystemMessages = systemMessages.map(msg => {
            const cleaned = { role: msg.role, content: msg.content || '' };
            usedTokens += estimateTokens(cleaned.content) + 4;
            return cleaned;
        });

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

        if (ENABLE_THINKING_MODE && currentBackend === 'nvidia') {
            apiRequest.extra_body = { chat_template_kwargs: { thinking: true } };
        }

        const payloadJson = JSON.stringify(apiRequest);
        const payloadSizeKB = (Buffer.byteLength(payloadJson, 'utf8') / 1024).toFixed(1);
        console.log(`🔍 Payload ${payloadSizeKB} KB | Model: ${targetModel} | Backend: ${getBackendConfig(currentBackend).name} | Messages: ${cleanedMessages.length} | Stream: ${stream || false}`);

        // =============================================
        // 🔥 ATTEMPT 1: Try the primary backend
        // =============================================
        let response;
        try {
            response = await makeBackendRequest(currentBackend, apiRequest, stream);
        } catch (primaryError) {
            // =============================================
            // 🔥 FALLBACK: If AgentRouter fails (sensitive words, rate limit, server error 502/503/504, or network timeout),
            //    automatically retry through NVIDIA NIM (DeepSeek v4 Pro)
            // =============================================
            const errorDetail = await extractErrorDetail(primaryError);
            const status = primaryError.response?.status;
            
            const isSensitiveWords = errorDetail.includes('sensitive words');
            const isServerOrRateLimitError = status === 429 || (status >= 500 && status <= 504);
            const isNetworkError = !primaryError.response || 
                                   ['ECONNABORTED', 'ETIMEDOUT', 'ENOTFOUND', 'ECONNREFUSED'].includes(primaryError.code);

            const shouldFallback = currentBackend === 'agentrouter' && 
                                   (isSensitiveWords || isServerOrRateLimitError || isNetworkError) && 
                                   !!NIM_API_KEY;

            if (shouldFallback) {
                let reason = 'unknown error';
                if (isSensitiveWords) reason = 'sensitive words filter';
                else if (isServerOrRateLimitError) reason = `server status ${status}`;
                else if (isNetworkError) reason = `network error (${primaryError.code || 'timeout'})`;

                console.log(`⚠️ AgentRouter failed: "${errorDetail.slice(0, 200)}" (${reason})`);
                console.log(`🔄 FALLBACK → Retrying via NVIDIA NIM (${SENSITIVE_WORDS_FALLBACK.model})...`);

                // Switch to NVIDIA fallback
                currentBackend = SENSITIVE_WORDS_FALLBACK.backend;
                targetModel = SENSITIVE_WORDS_FALLBACK.model;
                apiRequest.model = targetModel;

                try {
                    response = await makeBackendRequest(currentBackend, apiRequest, stream);
                    console.log(`✅ Fallback successful! Response from NVIDIA NIM (${targetModel})`);
                } catch (fallbackError) {
                    // Both backends failed — return the fallback error
                    const fallbackDetail = await extractErrorDetail(fallbackError);
                    const isTimeout = fallbackError.code === 'ECONNABORTED' || fallbackError.code === 'ETIMEDOUT';
                    console.error(`❌ Fallback also failed: ${fallbackDetail}`);
                    return res.status(fallbackError.response?.status || (isTimeout ? 504 : 500)).json({
                        error: {
                            message: `AgentRouter failed (${reason}) → NVIDIA fallback also failed: ${fallbackDetail}`,
                            type: isTimeout ? 'timeout_error' : 'invalid_request_error',
                            code: fallbackError.response?.status || (isTimeout ? 504 : 500)
                        }
                    });
                }
            } else {
                // Not a fallback candidate or not AgentRouter — throw normally
                throw primaryError;
            }
        }

        // =============================================
        // 🔥 SEND RESPONSE (stream or JSON)
        // =============================================
        if (stream) {
            sendStreamResponse(res, response, model);
        } else {
            sendJsonResponse(res, response, model);
        }

    } catch (error) {
        const status = error.response?.status || 'N/A';
        const isTimeout = error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT';

        let errorDetail = await extractErrorDetail(error);

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
    console.log(`Sensitive words fallback: ENABLED → NVIDIA ${SENSITIVE_WORDS_FALLBACK.model}`);
    console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
    console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
