async function test() {
    try {
        const res = await fetch('https://integrate.api.nvidia.com/v1/models', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await res.json();
        const models = data.data.map(m => m.id);
        const llama = models.filter(m => m.includes('llama-3.1') || m.includes('llama'));
        console.log("Llama Models:", llama);
    } catch (e) {
        console.error("Error:", e.message);
    }
}
test();
