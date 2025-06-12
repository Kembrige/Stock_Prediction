document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    form.addEventListener("submit", function () {
        const submitButton = form.querySelector("button[type='submit']");
        submitButton.innerHTML = "Loading...";
        submitButton.disabled = true;
    });
});