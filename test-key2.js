async function test() {
    const apiKey = "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";
    const models = [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "z-ai/glm5",
        "deepseek-ai/deepseek-v3.2"
    ];
    for (const model of models) {
        try {
            const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model,
                    messages: [{"role":"user","content":"Hi"}]
                })
            });
            const data = await res.text();
            console.log(model, "-> Status:", res.status, data.slice(0, 100));
        } catch(e) { console.error(e.message); }
    }
}
test();
