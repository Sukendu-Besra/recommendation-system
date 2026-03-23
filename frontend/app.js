async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) {
    const detail = data?.detail ? data.detail : res.statusText;
    throw new Error(detail);
  }
  return data;
}

function renderRecommendations(recommendations) {
  const ul = document.getElementById("results");
  ul.innerHTML = "";

  if (!recommendations || recommendations.length === 0) {
    ul.innerHTML = `
      <div class="empty-state">
        <i class='bx bx-ghost'></i>
        <p>No recommendations found.</p>
        <span class="subtext">Try different parameters or another user ID.</span>
      </div>
    `;
    return;
  }

  recommendations.forEach((r, index) => {
    const li = document.createElement("li");
    li.className = "result-item";
    li.style.animationDelay = `${index * 0.05}s`;
    
    const title = r.title ? r.title : "Unknown Title";
    const score = typeof r.score === "number" ? r.score.toFixed(4) : String(r.score);
    
    li.innerHTML = `
      <div class="result-rank">#${index + 1}</div>
      <div class="result-content">
        <div class="result-title">${title}</div>
        <div class="result-meta">
          <span class="result-id"><i class='bx bx-hash'></i> ${r.item_id}</span>
        </div>
      </div>
      <div class="result-score"><i class='bx bx-target-lock'></i> ${score}</div>
    `;
    ul.appendChild(li);
  });
}

function setStatus(msg, state = "active") {
  const el = document.getElementById("status");
  el.textContent = msg;
  
  // Re-apply base classes and dynamic state class
  el.className = `status-badge status-${state}`;
}

document.getElementById("recommendBtn").addEventListener("click", async () => {
  try {
    const userId = document.getElementById("userId").value;
    const strategy = document.getElementById("strategy").value;
    const k = document.getElementById("k").value;

    setStatus("Loading...", "active");
    const data = await fetchJson(
      `/recommend?user_id=${encodeURIComponent(userId)}&strategy=${encodeURIComponent(
        strategy
      )}&k=${encodeURIComponent(k)}&exclude_seen=true`
    );

    renderRecommendations(data.recommendations);
    setStatus("Success", "success");
    
    // Reset status after a few seconds
    setTimeout(() => {
      setStatus("Ready", "idle");
    }, 3000);
  } catch (err) {
    setStatus(`Error`, "error");
    console.error(err);
  }
});

document.getElementById("simulateBtn").addEventListener("click", async () => {
  try {
    const userId = document.getElementById("userId").value;
    const itemId = document.getElementById("simItemId").value;
    const rating = document.getElementById("simRating").value;
    const strategy = document.getElementById("simStrategy").value;

    setStatus("Simulating...", "active");
    const payload = await fetchJson("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: Number(userId),
        item_id: Number(itemId),
        rating: Number(rating),
        strategy: strategy,
        k: Number(document.getElementById("k").value),
      }),
    });

    renderRecommendations(payload.recommendations);
    setStatus("Updated", "success");
    
     // Reset status after a few seconds
    setTimeout(() => {
      setStatus("Ready", "idle");
    }, 3000);
  } catch (err) {
    setStatus(`Error`, "error");
    console.error(err);
  }
});

document.getElementById("resetBtn").addEventListener("click", async () => {
  try {
    setStatus("Resetting...", "active");
    await fetchJson("/reset_demo", { method: "POST" });
    setStatus("Reset OK", "success");
    document.getElementById("results").innerHTML = `
      <div class="empty-state">
        <i class='bx bx-reset'></i>
        <p>Overlay Cleared</p>
        <span class="subtext">Run recommend again to see default results.</span>
      </div>
    `;
    
    // Reset status after a few seconds
    setTimeout(() => {
      setStatus("Ready", "idle");
    }, 3000);
  } catch (err) {
    setStatus(`Error`, "error");
    console.error(err);
  }
});
