"""Minimal built-in web UI for the humanizer API.

The page is plain HTML, CSS, and JavaScript so it needs no template engine, build
step, or CDN. It calls the same ``/humanize`` endpoint, which means it works both
on localhost and through a temporary Colab tunnel.
"""

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Humanizer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --card: #ffffff;
      --ink: #17212b;
      --muted: #5f6b7a;
      --line: #d9e0e7;
      --accent: #155eef;
      --accent-ink: #ffffff;
      --danger-bg: #fff0f0;
      --danger-line: #f0b4b4;
      --danger-ink: #9d1c1c;
      --ok: #16794b;
      --radius: 14px;
      --shadow: 0 10px 30px rgba(23, 33, 43, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, #dce8ff 0, transparent 38%),
        var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
      line-height: 1.5;
    }

    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }

    header { margin-bottom: 22px; }
    h1 { margin: 0 0 6px; font-size: clamp(1.7rem, 4vw, 2.35rem); letter-spacing: -0.02em; }
    .subtitle { margin: 0; color: var(--muted); max-width: 65ch; }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
    }

    .field { margin-bottom: 16px; }
    label { display: block; font-weight: 650; margin-bottom: 6px; font-size: 0.94rem; }
    .hint { margin: 6px 0 0; color: var(--muted); font-size: 0.82rem; }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
      outline: none;
    }

    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(21, 94, 239, 0.14);
    }

    textarea { resize: vertical; min-height: 210px; }

    .row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .counter { color: var(--muted); font-size: 0.82rem; }

    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 15px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      background: #eef2f7;
      color: var(--ink);
      transition: filter 120ms ease, transform 120ms ease;
    }

    button:hover:not(:disabled) { filter: brightness(0.96); }
    button:active:not(:disabled) { transform: translateY(1px); }
    button:disabled { cursor: not-allowed; opacity: 0.6; }

    .primary {
      width: 100%;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 12px 18px;
      font-size: 1rem;
    }

    .ghost { background: transparent; border: 1px solid var(--line); }
    .small { padding: 6px 10px; font-size: 0.84rem; }

    .error {
      margin-top: 14px;
      padding: 12px 14px;
      border: 1px solid var(--danger-line);
      border-radius: 10px;
      background: var(--danger-bg);
      color: var(--danger-ink);
      white-space: pre-wrap;
      font-size: 0.9rem;
    }

    .result-head { margin: 0 0 12px; font-size: 1.05rem; }

    pre {
      min-height: 254px;
      margin: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fbfcfe;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92rem;
      line-height: 1.65;
    }

    .meta {
      margin-top: 10px;
      min-height: 1.2em;
      color: var(--muted);
      font-size: 0.82rem;
    }

    .meta.ok { color: var(--ok); }

    .footer {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.82rem;
    }

    @media (max-width: 820px) {
      .layout { grid-template-columns: 1fr; }
      .shell { padding-top: 22px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>AI Humanizer</h1>
      <p class="subtitle">
        Rewrite English text for clarity, fluency, and readability while preserving
        meaning. The model runs in the same API process.
      </p>
    </header>

    <div class="layout">
      <section class="card" aria-labelledby="input-heading">
        <h2 id="input-heading" class="result-head">Your text</h2>

        <form id="humanize-form">
          <div class="field">
            <label for="api-key">API key</label>
            <input
              id="api-key"
              type="password"
              autocomplete="off"
              spellcheck="false"
              placeholder="Paste the key printed by the Colab notebook"
            >
            <p class="hint">
              Leave this empty only when the server has no <code>API_KEY</code> set.
              The key is sent only to this server.
            </p>
          </div>

          <div class="field">
            <label for="text">Text to rewrite</label>
            <textarea
              id="text"
              placeholder="Paste a paragraph here..."
              required
            ></textarea>
            <div class="row" style="margin-top: 6px;">
              <span id="counter" class="counter">0 characters</span>
              <button type="button" id="sample" class="ghost small">Use sample</button>
            </div>
          </div>

          <div class="field">
            <label for="style">Writing style</label>
            <select id="style">
              <option value="natural">Natural</option>
              <option value="academic">Academic</option>
              <option value="professional">Professional</option>
              <option value="simple">Simple</option>
              <option value="friendly">Friendly</option>
            </select>
          </div>

          <button type="submit" id="submit" class="primary">Humanize</button>
          <div id="error" class="error" role="alert" hidden></div>
        </form>
      </section>

      <section class="card" aria-labelledby="output-heading">
        <div class="row" style="margin-bottom: 12px;">
          <h2 id="output-heading" class="result-head" style="margin: 0;">Result</h2>
          <button type="button" id="copy" class="ghost small" disabled>Copy</button>
        </div>
        <pre id="output" aria-live="polite">Your rewritten text will appear here.</pre>
        <div id="meta" class="meta"></div>
      </section>
    </div>

    <p class="footer">
      Review important output manually. This tool improves writing quality; it does
      not guarantee a particular AI-detector result.
    </p>
  </main>

  <script>
    const form = document.querySelector("#humanize-form");
    const apiKey = document.querySelector("#api-key");
    const text = document.querySelector("#text");
    const style = document.querySelector("#style");
    const submit = document.querySelector("#submit");
    const sample = document.querySelector("#sample");
    const counter = document.querySelector("#counter");
    const errorBox = document.querySelector("#error");
    const output = document.querySelector("#output");
    const meta = document.querySelector("#meta");
    const copy = document.querySelector("#copy");

    const SAMPLE = "Technology plays an important role in modern education. It gives students access to more information, but it also requires teachers to guide them carefully. In 2024, many schools began testing new tools in the classroom.";

    sample.addEventListener("click", () => {
      text.value = SAMPLE;
      updateCounter();
      text.focus();
    });

    text.addEventListener("input", updateCounter);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await humanize();
    });

    text.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        humanize();
      }
    });

    copy.addEventListener("click", async () => {
      const value = output.textContent;
      if (!value.trim() || value === "Your rewritten text will appear here.") return;

      try {
        await navigator.clipboard.writeText(value);
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1400);
      } catch {
        const range = document.createRange();
        range.selectNodeContents(output);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        copy.textContent = "Select and copy";
      }
    });

    function updateCounter() {
      const count = text.value.length;
      counter.textContent = count === 1 ? "1 character" : `${count} characters`;
    }

    function clearError() {
      errorBox.hidden = true;
      errorBox.textContent = "";
    }

    function showError(message) {
      errorBox.hidden = false;
      errorBox.textContent = message;
    }

    function formatDetail(detail) {
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return detail.map((item) => {
          const location = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
          const prefix = location ? `${location}: ` : "";
          return `${prefix}${item.msg || "Invalid value"}`;
        }).join("\n");
      }
      return "The request failed. Check the server log.";
    }

    async function humanize() {
      clearError();

      if (!text.value.trim()) {
        showError("Enter some text first.");
        text.focus();
        return;
      }

      submit.disabled = true;
      submit.textContent = "Rewriting...";
      meta.textContent = "";
      meta.className = "meta";

      const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
      };
      if (apiKey.value.trim()) {
        headers["X-API-Key"] = apiKey.value.trim();
      }

      try {
        const response = await fetch("/humanize", {
          method: "POST",
          headers,
          body: JSON.stringify({
            text: text.value,
            style: style.value
          })
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
          showError(`HTTP ${response.status}\n${formatDetail(data.detail)}`);
          return;
        }

        output.textContent = data.humanized_text;
        copy.disabled = false;
        meta.className = "meta ok";
        meta.textContent = `${data.style} | ${data.original_characters} to ${data.rewritten_characters} characters | HTTP ${response.status}`;
      } catch (error) {
        showError(
          "Could not reach the API. Confirm the server is running and the URL is correct.\n" +
          String(error)
        );
      } finally {
        submit.disabled = false;
        submit.textContent = "Humanize";
      }
    }

    updateCounter();
  </script>
</body>
</html>
"""
