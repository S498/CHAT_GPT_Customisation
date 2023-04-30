const getElement = (selector) => document.querySelector(selector);
const promptInput = getElement('#prompt-input');
const askGPTButton = getElement('#ask-gpt-button');
const responseDiv = getElement('#response');

const generateResponse = (prompt) => {
    axios.post('/chatgpt', { prompt })
        .then(({ data }) => responseDiv.textContent = data.response)
        .catch((error) => console.error(error));
};

askGPTButton.addEventListener('click', () => {
    const prompt = promptInput.value.trim();
    if (prompt) {
        askGPTButton.classList.add('blink');
        generateResponse(prompt);
    }
});
