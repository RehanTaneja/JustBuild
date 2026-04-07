from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ArchitecturePlan, ProductSpecification

"""
This file is responsible for generating a static web app for the prototype.
"""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "prototype"

# Renders index.html which includes product summary API contracts, features, user stories etc. 
def render_index_html(spec: ProductSpecification, architecture: ArchitecturePlan) -> str:
    feature_items = "\n".join(f"<li>{feature}</li>" for feature in spec.features)
    user_story_items = "\n".join(f"<li>{story}</li>" for story in spec.user_stories)
    api_items = "\n".join(f"<li><code>{contract}</code></li>" for contract in spec.api_contracts)
    assumption_items = "\n".join(f"<li>{item}</li>" for item in spec.assumptions)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{spec.title}</title>
  <link rel="stylesheet" href="./styles.css" />
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">Autonomous Prototype</p>
      <h1>{spec.title}</h1>
      <p class="lede">{spec.product_summary}</p>
      <div class="hero-card">
        <h2>Primary Workflow</h2>
        <form id="idea-form">
          <label for="idea-input">Describe the request</label>
          <textarea id="idea-input" rows="5" placeholder="Enter a use case or customer request"></textarea>
          <button type="submit">Generate prototype response</button>
        </form>
        <div class="result" id="result-panel" aria-live="polite"></div>
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Feature Breakdown</h2>
        <ul>{feature_items}</ul>
      </article>
      <article class="panel">
        <h2>User Stories</h2>
        <ul>{user_story_items}</ul>
      </article>
      <article class="panel">
        <h2>API Contracts</h2>
        <ul>{api_items}</ul>
      </article>
      <article class="panel">
        <h2>Architecture Highlights</h2>
        <p>{architecture.summary}</p>
        <ul>{assumption_items}</ul>
      </article>
    </section>

    <section class="panel history">
      <div class="section-head">
        <h2>Recent Generated Items</h2>
        <span>Mock session storage</span>
      </div>
      <div id="history-list"></div>
    </section>
  </main>
  <script src="./app.js"></script>
</body>
</html>
"""

# builds UI styling
def render_styles_css() -> str:
    return """\
:root {
  --bg: #f4efe6;
  --surface: rgba(255, 252, 247, 0.86);
  --text: #1f1e1c;
  --muted: #615c53;
  --accent: #0d6f63;
  --accent-strong: #0a524a;
  --border: rgba(31, 30, 28, 0.12);
  --shadow: 0 30px 80px rgba(52, 43, 31, 0.12);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(13, 111, 99, 0.15), transparent 25%),
    radial-gradient(circle at top right, rgba(184, 104, 56, 0.12), transparent 22%),
    linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
  min-height: 100vh;
}

.shell {
  width: min(1120px, calc(100% - 2rem));
  margin: 0 auto;
  padding: 3rem 0 4rem;
}

.hero {
  display: grid;
  gap: 1rem;
  margin-bottom: 2rem;
}

.eyebrow {
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  font-size: 0.8rem;
  margin: 0;
}

h1, h2 {
  margin: 0;
}

h1 {
  font-size: clamp(2.5rem, 6vw, 4.8rem);
  line-height: 0.95;
}

.lede {
  max-width: 56rem;
  color: var(--muted);
  font-size: 1.1rem;
}

.hero-card,
.panel {
  background: var(--surface);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow);
}

.hero-card {
  padding: 1.5rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.panel {
  padding: 1.25rem;
}

label,
span,
li,
p {
  color: var(--muted);
}

textarea {
  width: 100%;
  border-radius: 16px;
  border: 1px solid var(--border);
  padding: 1rem;
  margin: 0.75rem 0 1rem;
  font: inherit;
  background: rgba(255, 255, 255, 0.75);
}

button {
  border: 0;
  border-radius: 999px;
  padding: 0.9rem 1.35rem;
  background: var(--accent);
  color: white;
  font: inherit;
  cursor: pointer;
  transition: transform 180ms ease, background 180ms ease;
}

button:hover {
  transform: translateY(-1px);
  background: var(--accent-strong);
}

.result,
.history-item {
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid var(--border);
}

.section-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
}

@media (max-width: 700px) {
  .shell {
    width: min(100% - 1rem, 100%);
    padding-top: 1rem;
  }

  h1 {
    font-size: 2.5rem;
  }
}
"""

# Render app.js which is fake AI response generator, simulates user interaction basically.
def render_app_js(spec: ProductSpecification) -> str:
    serialized = json.dumps(
        {
            "title": spec.title,
            "features": spec.features,
            "stories": spec.user_stories,
        },
        indent=2,
    )
    return f"""\
const prototypeMeta = {serialized};

const form = document.getElementById("idea-form");
const ideaInput = document.getElementById("idea-input");
const resultPanel = document.getElementById("result-panel");
const historyList = document.getElementById("history-list");

const history = [];

function renderHistory() {{
  historyList.innerHTML = "";
  history.slice().reverse().forEach((item) => {{
    const node = document.createElement("article");
    node.className = "history-item";
    node.innerHTML = `
      <strong>${{item.name}}</strong>
      <p>${{item.summary}}</p>
      <small>Status: ${{item.status}}</small>
    `;
    historyList.appendChild(node);
  }});
}}

form.addEventListener("submit", (event) => {{
  event.preventDefault();
  const rawValue = ideaInput.value.trim();
  if (!rawValue) {{
    resultPanel.textContent = "Add a short request so the prototype can generate a response.";
    return;
  }}

  const summary = `Generated a ${{
    prototypeMeta.title
  }} concept response for: "${{rawValue}}" using ${{
    prototypeMeta.features.length
  }} scaffolded features.`;

  history.push({{
    name: rawValue.slice(0, 40),
    status: "drafted",
    summary,
  }});

  resultPanel.innerHTML = `
    <h3>Generated Response</h3>
    <p>${{summary}}</p>
    <p>Next recommended step: validate assumptions, wire a real API, and persist the item state.</p>
  `;
  ideaInput.value = "";
  renderHistory();
}});
"""

# writes them all together
def write_prototype_files(base_dir: Path, spec: ProductSpecification, architecture: ArchitecturePlan) -> list[Path]:
    prototype_dir = base_dir / "prototype"
    prototype_dir.mkdir(parents=True, exist_ok=True)

    files = {
        prototype_dir / "index.html": render_index_html(spec, architecture),
        prototype_dir / "styles.css": render_styles_css(),
        prototype_dir / "app.js": render_app_js(spec),
        prototype_dir / "README.md": (
            f"# {spec.title}\n\n"
            "This browser prototype was generated by the JustBuild multi-agent system.\n\n"
            "Open `index.html` in a browser to interact with the generated prototype.\n"
        ),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    return list(files.keys())
