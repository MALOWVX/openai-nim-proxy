async function test() {
    try {
        const apiKey = "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";
        const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "z-ai/glm4.7",
                messages: [{"role":"user","content":"Hi"}]
            })
        });
        console.log("Status:", res.status);
        const data = await res.text();
        console.log("Data:", data);
    } catch(e) { console.error(e); }
}
test();
