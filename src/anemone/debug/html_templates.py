"""Static HTML templates for browser-based debug trace replay."""

from __future__ import annotations


def render_replay_index_html() -> str:
    """Return the self-contained replay viewer HTML document."""
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Anemone Debug Replay</title>
    <style>
      :root {
        --bg: #f3ede2;
        --panel: #fffaf3;
        --panel-strong: #f5ebdc;
        --ink: #251a12;
        --muted: #715e50;
        --accent: #204a40;
        --accent-soft: #d8e6df;
        --border: #d5c6b4;
        --shadow: 0 18px 48px rgba(37, 26, 18, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background:
          radial-gradient(circle at top left, rgba(255, 255, 255, 0.75), transparent 28rem),
          linear-gradient(160deg, #f8f1e7 0%, #ece2d4 100%);
        color: var(--ink);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      }

      .shell {
        display: grid;
        grid-template-columns: minmax(18rem, 24rem) 1fr;
        min-height: 100vh;
      }

      .timeline-pane,
      .detail-pane {
        padding: 1.25rem;
      }

      .timeline-pane {
        border-right: 1px solid var(--border);
        background: rgba(255, 250, 243, 0.92);
        backdrop-filter: blur(8px);
      }

      .detail-pane {
        display: grid;
        gap: 1rem;
      }

      .card {
        background: var(--panel);
        border: 1px solid rgba(113, 94, 80, 0.18);
        border-radius: 1rem;
        box-shadow: var(--shadow);
      }

      .heading {
        margin: 0 0 0.4rem;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
      }

      .subheading,
      .meta,
      .timeline-summary,
      .snapshot-note {
        color: var(--muted);
      }

      .timeline-header,
      .detail-header,
      .controls,
      .snapshot-header {
        padding: 1rem 1rem 0;
      }

      .timeline-list {
        list-style: none;
        margin: 0;
        padding: 0.75rem;
        display: grid;
        gap: 0.55rem;
        max-height: calc(100vh - 8rem);
        overflow: auto;
      }

      .timeline-item {
        width: 100%;
        border: 1px solid transparent;
        border-radius: 0.85rem;
        background: var(--panel-strong);
        color: inherit;
        cursor: pointer;
        text-align: left;
        padding: 0.8rem 0.9rem;
        font: inherit;
        transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      }

      .timeline-item:hover {
        transform: translateY(-1px);
        border-color: rgba(32, 74, 64, 0.25);
      }

      .timeline-item.is-selected {
        background: var(--accent);
        border-color: var(--accent);
        color: #f6f1e8;
      }

      .timeline-line {
        display: block;
        font-family: "IBM Plex Mono", "Fira Code", "SFMono-Regular", monospace;
        font-size: 0.88rem;
      }

      .timeline-summary {
        display: block;
        margin-top: 0.35rem;
        font-size: 0.86rem;
      }

      .timeline-item.is-selected .timeline-summary,
      .timeline-item.is-selected .meta,
      .timeline-item.is-selected .snapshot-note {
        color: rgba(246, 241, 232, 0.8);
      }

      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
      }

      .controls button {
        border: 1px solid var(--accent);
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font: inherit;
        cursor: pointer;
      }

      .controls button:disabled {
        cursor: not-allowed;
        opacity: 0.45;
      }

      .detail-card,
      .snapshot-card {
        padding: 0 1rem 1rem;
      }

      .detail-grid {
        display: grid;
        gap: 0.55rem;
      }

      .detail-label {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .detail-value {
        margin: 0;
        font-family: "IBM Plex Mono", "Fira Code", "SFMono-Regular", monospace;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .snapshot-stage {
        min-height: 26rem;
        border: 1px dashed rgba(32, 74, 64, 0.3);
        border-radius: 0.85rem;
        background:
          linear-gradient(135deg, rgba(216, 230, 223, 0.55), rgba(255, 250, 243, 0.92));
        display: grid;
        place-items: center;
        padding: 1rem;
      }

      .snapshot-stage img {
        max-width: 100%;
        max-height: 70vh;
        display: block;
        box-shadow: 0 12px 36px rgba(37, 26, 18, 0.18);
        background: white;
      }

      .snapshot-stage pre {
        width: 100%;
        margin: 0;
        overflow: auto;
        white-space: pre-wrap;
        font-family: "IBM Plex Mono", "Fira Code", "SFMono-Regular", monospace;
        font-size: 0.84rem;
      }

      .empty {
        color: var(--muted);
        font-style: italic;
      }

      @media (max-width: 900px) {
        .shell {
          grid-template-columns: 1fr;
        }

        .timeline-pane {
          border-right: none;
          border-bottom: 1px solid var(--border);
        }

        .timeline-list {
          max-height: 18rem;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="timeline-pane">
        <section class="card">
          <div class="timeline-header">
            <h1 class="heading">Debug Trace Replay</h1>
            <p class="subheading" id="timeline-meta">Loading trace.json...</p>
          </div>
          <ol class="timeline-list" id="timeline"></ol>
        </section>
      </aside>
      <main class="detail-pane">
        <section class="card">
          <div class="controls">
            <button id="previous-entry" type="button">Previous</button>
            <button id="next-entry" type="button">Next</button>
            <span class="meta" id="current-index">No entry selected</span>
          </div>
          <div class="detail-header">
            <h2 class="heading" id="entry-title">Replay details</h2>
          </div>
          <div class="detail-card detail-grid">
            <div>
              <div class="detail-label">Event Type</div>
              <p class="detail-value" id="entry-type">-</p>
            </div>
            <div>
              <div class="detail-label">Event Summary</div>
              <p class="detail-value" id="entry-summary">-</p>
            </div>
          </div>
        </section>
        <section class="card">
          <div class="snapshot-header">
            <h2 class="heading">Nearest Snapshot</h2>
            <p class="snapshot-note" id="snapshot-status">No snapshot loaded</p>
          </div>
          <div class="snapshot-card">
            <div class="snapshot-stage" id="snapshot-view">
              <div class="empty">No snapshot available yet</div>
            </div>
          </div>
        </section>
      </main>
    </div>
    <script>
      const timelineElement = document.getElementById("timeline");
      const timelineMetaElement = document.getElementById("timeline-meta");
      const previousButton = document.getElementById("previous-entry");
      const nextButton = document.getElementById("next-entry");
      const currentIndexElement = document.getElementById("current-index");
      const entryTitleElement = document.getElementById("entry-title");
      const entryTypeElement = document.getElementById("entry-type");
      const entrySummaryElement = document.getElementById("entry-summary");
      const snapshotStatusElement = document.getElementById("snapshot-status");
      const snapshotViewElement = document.getElementById("snapshot-view");

      let entries = [];
      let selectedIndex = 0;

      function formatIndex(index) {
        return String(index).padStart(4, "0");
      }

      function nearestSnapshotAtOrBefore(index) {
        for (let current = index; current >= 0; current -= 1) {
          const entry = entries[current];
          if (entry && entry.snapshot_file) {
            return entry;
          }
        }
        return null;
      }

      function renderTimeline() {
        timelineElement.replaceChildren();

        if (entries.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "No replay entries available.";
          timelineElement.appendChild(emptyItem);
          return;
        }

        entries.forEach((entry, position) => {
          const listItem = document.createElement("li");
          const button = document.createElement("button");
          button.type = "button";
          button.className = "timeline-item";
          if (position === selectedIndex) {
            button.classList.add("is-selected");
          }

          const line = document.createElement("span");
          line.className = "timeline-line";
          line.textContent = `[${formatIndex(entry.index)}] ${entry.event_summary}`;

          const summary = document.createElement("span");
          summary.className = "timeline-summary";
          summary.textContent = entry.has_snapshot
            ? `${entry.event_type} | snapshot attached`
            : `${entry.event_type} | no snapshot`;

          button.appendChild(line);
          button.appendChild(summary);
          button.addEventListener("click", () => {
            selectEntry(position);
          });

          listItem.appendChild(button);
          timelineElement.appendChild(listItem);
        });
      }

      function updateControls() {
        const hasEntries = entries.length > 0;
        previousButton.disabled = !hasEntries || selectedIndex <= 0;
        nextButton.disabled = !hasEntries || selectedIndex >= entries.length - 1;
      }

      function renderEmptyDetails(message) {
        currentIndexElement.textContent = "No entry selected";
        entryTitleElement.textContent = "Replay details";
        entryTypeElement.textContent = "-";
        entrySummaryElement.textContent = message;
        snapshotStatusElement.textContent = "No snapshot loaded";
        snapshotViewElement.replaceChildren();
        const emptyMessage = document.createElement("div");
        emptyMessage.className = "empty";
        emptyMessage.textContent = message;
        snapshotViewElement.appendChild(emptyMessage);
        updateControls();
      }

      async function renderSnapshot(entryIndex) {
        const snapshotEntry = nearestSnapshotAtOrBefore(entryIndex);
        snapshotViewElement.replaceChildren();

        if (!snapshotEntry) {
          snapshotStatusElement.textContent = "No snapshot available yet";
          const emptyMessage = document.createElement("div");
          emptyMessage.className = "empty";
          emptyMessage.textContent = "No snapshot available yet";
          snapshotViewElement.appendChild(emptyMessage);
          return;
        }

        const sourceLabel = snapshotEntry.index === entryIndex
          ? `Snapshot attached to entry [${formatIndex(snapshotEntry.index)}]`
          : `Using nearest prior snapshot from entry [${formatIndex(snapshotEntry.index)}]`;
        snapshotStatusElement.textContent = sourceLabel;

        if (snapshotEntry.snapshot_file.endsWith(".dot")) {
          const response = await fetch(snapshotEntry.snapshot_file);
          const dotSource = await response.text();
          const block = document.createElement("pre");
          block.textContent = dotSource;
          snapshotViewElement.appendChild(block);
          return;
        }

        const image = document.createElement("img");
        image.alt = `Snapshot for entry ${snapshotEntry.index}`;
        image.src = snapshotEntry.snapshot_file;
        snapshotViewElement.appendChild(image);
      }

      async function renderDetails() {
        if (entries.length === 0) {
          renderEmptyDetails("No replay entries available.");
          return;
        }

        const entry = entries[selectedIndex];
        currentIndexElement.textContent = `Entry ${selectedIndex + 1} of ${entries.length}`;
        entryTitleElement.textContent = `[${formatIndex(entry.index)}] ${entry.event_summary}`;
        entryTypeElement.textContent = entry.event_type;
        entrySummaryElement.textContent = entry.event_summary;
        updateControls();
        await renderSnapshot(selectedIndex);
      }

      async function selectEntry(index) {
        selectedIndex = index;
        renderTimeline();
        await renderDetails();
      }

      async function loadTrace() {
        try {
          const response = await fetch("trace.json");
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }

          const payload = await response.json();
          entries = Array.isArray(payload.entries) ? payload.entries : [];
          timelineMetaElement.textContent = `${payload.entry_count ?? entries.length} entries loaded`;

          renderTimeline();
          if (entries.length > 0) {
            await selectEntry(0);
          } else {
            renderEmptyDetails("No replay entries available.");
          }
        } catch (error) {
          timelineMetaElement.textContent = "Failed to load trace.json";
          renderEmptyDetails(`Unable to load replay payload: ${error}`);
        }
      }

      previousButton.addEventListener("click", async () => {
        if (selectedIndex > 0) {
          await selectEntry(selectedIndex - 1);
        }
      });

      nextButton.addEventListener("click", async () => {
        if (selectedIndex < entries.length - 1) {
          await selectEntry(selectedIndex + 1);
        }
      });

      void loadTrace();
    </script>
  </body>
</html>
"""


__all__ = ["render_replay_index_html"]
