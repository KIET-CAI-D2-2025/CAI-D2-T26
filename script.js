document.getElementById("predict-btn").addEventListener("click", function () {
    let age = document.getElementById("age").value;
    let income = document.getElementById("income").value;
    let gender = document.querySelector('input[name="gender"]:checked').value;
    let smoke = document.querySelector('input[name="smoke"]:checked').value;
    let health = document.getElementById("health").value;

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age, income, gender, smoke, health })
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        resultDiv.style.background = data.eligible ? "green" : "red";
        
        let message = data.message;
        if (data.eligible && data.policies) {
            message += '<br><br>Available Policies:<br>';
            data.policies.forEach(policy => {
                const premium = data.premiums[policy].toFixed(2);
                message += `${policy}: $${premium} per month<br>`;
            });
        }
        
        resultDiv.innerHTML = message;
        resultDiv.classList.remove("hidden");
    })
    .catch(error => {
        console.error('Error:', error);
        let resultDiv = document.getElementById("result");
        resultDiv.style.background = "red";
        resultDiv.innerHTML = "An error occurred. Please try again.";
        resultDiv.classList.remove("hidden");
    });
});

function updateAgeValue(val) {
    document.getElementById("age-value").textContent = val;
}