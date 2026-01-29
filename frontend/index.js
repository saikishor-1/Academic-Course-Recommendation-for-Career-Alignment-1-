

document.getElementById("form").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        CGPA: Number(CGPA.value),
        Python: Number(Python.value),
        Machine_Learning: Number(Machine_Learning.value),
        SQL: Number(SQL.value),
        Projects: Number(Projects.value),
        Interest_Domain: Interest_Domain.value,
        Career_Goal: Career_Goal.value,
        Industry_Demand_Score: Number(Industry_Demand_Score.value),
        Internship_Experience: Number(Internship_Experience.value)
    };

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();

    let output = "Top 3 Recommended Courses:<br>";
    result.forEach(r => {
        output += `${r.course} → ${r.confidence}%<br>`;
    });

    document.getElementById("result").innerHTML = output;
});
