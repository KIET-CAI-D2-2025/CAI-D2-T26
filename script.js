document.getElementById('predictForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    let age = document.getElementById('age').value;
    let gender = document.getElementById('gender').value;
    let income = document.getElementById('income').value;
    let health = document.getElementById('health').value;
    let smoke = document.getElementById('smoke').value;

    let response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age, gender, income, health, smoke })
    });

    let data = await response.json();

    document.getElementById('result').innerHTML = `<strong>${data.message}</strong>`;

    if (data.suggestions) {
        document.getElementById('suggestions').innerHTML = `💡 ${data.suggestions}`;
    } else {
        document.getElementById('suggestions').innerHTML = '';
    }

    if (data.eligible) {
        let policyText = `<strong>Eligible Policies:</strong> ${data.policies.join(", ")}`;
        document.getElementById('policies').innerHTML = policyText;

        let premiumText = "<strong>Estimated Premiums:</strong><br>";
        for (let policy in data.premiums) {
            premiumText += `${policy}: $${data.premiums[policy].toFixed(2)}<br>`;
        }
        document.getElementById('premiums').innerHTML = premiumText;
    } else {
        document.getElementById('policies').innerHTML = "";
        document.getElementById('premiums').innerHTML = "";
    }
});