async function test() {
    const models = [
        'meta/llama-3.1-70b-instruct',
        'z-ai/glm5',
        'deepseek-ai/deepseek-v3.2',
        'deepseek-ai/deepseek-v3.1',
        'z-ai/glm4.7',
        'deepseek-ai/deepseek-v3_1-terminus',
        'meta/llama-3.3-70b-instruct',
        'deepseek-ai/deepseek-r1'
    ];
    for (const model of models) {
        try {
            const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer fake-key`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model,
                    messages: [{"role":"user","content":"Hi"}]
                })
            });
            console.log(model, "->", res.status);
        } catch(e) { }
    }
}
test();
