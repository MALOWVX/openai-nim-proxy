async function test() {
    try {
        const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer fake-key`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "deepseek-ai/deepseek-r1",
                messages: [{"role":"user","content":"Hi"}]
            })
        });
        console.log("Status:", res.status);
        const data = await res.text();
        console.log("Data:", data);
    } catch (e) {
        console.error("Error:", e.message);
    }
}
test();
