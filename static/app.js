const faceForm = document.getElementById("face-search");
const faceResults = document.getElementById("face-results");

if (faceForm) {
  faceForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(faceForm);
    faceResults.innerHTML = "<p>Searching...</p>";

    try {
      const response = await fetch("/api/face-search", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        faceResults.innerHTML = `<p class="error">${data.error || "Search failed."}</p>`;
        return;
      }
      if (!data.results.length) {
        faceResults.innerHTML = "<p>No matches found.</p>";
        return;
      }
      faceResults.innerHTML = data.results
        .map(
          (result) => `
            <article class="card">
              <img src="/uploads/${result.filename}" alt="Match" />
              <div class="card-body">
                <p><strong>Similarity:</strong> ${result.score}</p>
              </div>
            </article>
          `
        )
        .join("");
    } catch (error) {
      faceResults.innerHTML = `<p class="error">${error.message}</p>`;
    }
  });
}
