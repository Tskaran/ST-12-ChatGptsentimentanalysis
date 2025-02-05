document.getElementById('sendButton').addEventListener('click', function() {
    const userInput = document.getElementById('userInput').value;
    const messages = document.getElementById('messages');
    
    if (userInput) {
        const messageElement = document.createElement('div');
        messageElement.textContent = `:User  ${userInput}`;
        messages.appendChild(messageElement);
        document.getElementById('userInput').value = '';
    }
});

document.getElementById('predictButton').addEventListener('click', function() {
    alert('Sentiment prediction feature is not implemented yet!');
});