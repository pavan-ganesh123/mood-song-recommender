document.addEventListener("DOMContentLoaded", function () {
    console.log("JavaScript Loaded!");

    // Fetch songs dynamically (if needed)
    fetch("/songs?mood=joy")
        .then(response => response.json())
        .then(data => {
            console.log("Songs Data:", data);
        })
        .catch(error => {
            console.error("Error fetching songs:", error);
        });
});
