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

      .controls select,
      .controls input {
        border: 1px solid rgba(32, 74, 64, 0.2);
        border-radius: 0.75rem;
        padding: 0.45rem 0.75rem;
        background: var(--panel);
        color: var(--ink);
        font: inherit;
      }

      .controls button:disabled {
        cursor: not-allowed;
        opacity: 0.45;
      }

      .checkbox-control {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--muted);
        font-size: 0.92rem;
      }

      .checkbox-control input {
        accent-color: var(--accent);
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

      .snapshot-stage svg {
        max-width: 100%;
        max-height: 70vh;
        display: block;
        box-shadow: 0 12px 36px rgba(37, 26, 18, 0.18);
        background: white;
      }

      .snapshot-stage svg g.node {
        cursor: pointer;
      }

      .snapshot-stage svg g.node.is-selected ellipse,
      .snapshot-stage svg g.node.is-selected polygon,
      .snapshot-stage svg g.node.is-selected path {
        stroke: #a53f1a;
        stroke-width: 3px;
      }

      .snapshot-stage svg g.node.is-selected text {
        font-weight: 700;
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

      .breakpoint-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 0.5rem;
      }

      .breakpoint-item {
        border: 1px solid rgba(32, 74, 64, 0.16);
        border-radius: 0.8rem;
        background: var(--panel-strong);
        padding: 0.7rem 0.85rem;
      }

      .node-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 0.45rem;
      }

      .node-list-item {
        width: 100%;
        text-align: left;
        border: 1px solid rgba(32, 74, 64, 0.16);
        border-radius: 0.8rem;
        background: var(--panel-strong);
        padding: 0.7rem 0.85rem;
        cursor: pointer;
      }

      .node-list-item.is-selected {
        border-color: rgba(165, 63, 26, 0.5);
        background: rgba(165, 63, 26, 0.08);
      }

      .node-list-label {
        display: block;
        font-weight: 600;
      }

      .node-list-summary {
        display: block;
        margin-top: 0.2rem;
        color: var(--muted);
      }

      .node-details-grid {
        display: grid;
        gap: 0.85rem;
      }

      .node-inspector-section {
        display: grid;
        gap: 0.45rem;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(32, 74, 64, 0.12);
        border-radius: 0.85rem;
        background: rgba(255, 250, 243, 0.55);
      }

      .node-details-list {
        margin: 0;
        padding-left: 1.1rem;
      }

      .node-details-pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--ink);
      }

      .node-details-field {
        display: grid;
        gap: 0.18rem;
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
            <p class="subheading" id="timeline-meta">Loading replay payload...</p>
            <div class="controls">
              <input id="timeline-search" type="text" placeholder="Search events or node ids">
              <select id="timeline-event-filter">
                <option value="all">All Events</option>
                <option value="SearchIterationStarted">Iteration Started</option>
                <option value="NodeSelected">Node Selected</option>
                <option value="NodeOpeningPlanned">Node Opening Planned</option>
                <option value="ChildLinked">Child Linked</option>
                <option value="DirectValueAssigned">Direct Value Assigned</option>
                <option value="BackupStarted">Backup Started</option>
                <option value="BackupFinished">Backup Finished</option>
                <option value="SearchIterationCompleted">Iteration Completed</option>
              </select>
              <input id="jump-entry-index" type="number" min="0" step="1" placeholder="Entry index">
              <button id="jump-entry-button" type="button">Go</button>
              <button id="jump-next-breakpoint" type="button">Next Breakpoint Hit</button>
              <button id="jump-next-pv-change" type="button">Next PV Change</button>
              <button id="jump-next-value-change" type="button">Next Value Change</button>
              <button id="jump-next-selected-node-event" type="button">Next Selected Node Event</button>
            </div>
            <p class="timeline-summary" id="timeline-filter-status">Showing 0 / 0 entries</p>
          </div>
          <ol class="timeline-list" id="timeline"></ol>
        </section>
      </aside>
      <main class="detail-pane">
        <section class="card">
          <div class="controls">
            <button id="previous-entry" type="button">Previous</button>
            <button id="next-entry" type="button">Next</button>
            <button id="pause-search" type="button">Pause</button>
            <button id="resume-search" type="button">Resume</button>
            <button id="step-search" type="button">Step</button>
            <label class="checkbox-control" for="auto-follow-latest">
              <input id="auto-follow-latest" type="checkbox" checked>
              <span>Auto-follow latest</span>
            </label>
            <span class="meta" id="current-index">No entry selected</span>
            <span class="meta" id="command-status">Live control unavailable</span>
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
        <section class="card">
          <div class="detail-header">
            <h2 class="heading">Node Inspection</h2>
            <p class="subheading" id="node-selection-status">No node selected</p>
          </div>
          <div class="detail-card">
            <ul class="node-list" id="node-list">
              <li class="empty">No snapshot nodes available yet.</li>
            </ul>
          </div>
          <div class="detail-card node-details-grid" id="node-details">
            <section class="node-inspector-section">
              <h3 class="heading">Identity</h3>
              <div class="node-details-field">
                <div class="detail-label">Node ID</div>
                <p class="detail-value" id="selected-node-id">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Depth</div>
                <p class="detail-value" id="selected-node-depth">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Root</div>
                <p class="detail-value" id="selected-node-root">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">State Tag</div>
                <p class="detail-value" id="selected-node-state-tag">-</p>
              </div>
            </section>
            <section class="node-inspector-section">
              <h3 class="heading">Evaluation</h3>
              <div class="node-details-field">
                <div class="detail-label">Direct Value</div>
                <p class="detail-value" id="selected-node-direct-value">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Backed-Up Value</div>
                <p class="detail-value" id="selected-node-backed-up-value">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Principal Variation</div>
                <p class="detail-value" id="selected-node-pv">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Over Event</div>
                <p class="detail-value" id="selected-node-over-event">-</p>
              </div>
            </section>
            <section class="node-inspector-section">
              <h3 class="heading">Structure</h3>
              <div class="node-details-field">
                <div class="detail-label">Parents</div>
                <p class="detail-value" id="selected-node-parents">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Children</div>
                <p class="detail-value" id="selected-node-children">-</p>
              </div>
              <div class="node-details-field">
                <div class="detail-label">Outgoing Edges</div>
                <ul class="node-details-list" id="selected-node-outgoing-edges">
                  <li class="empty">-</li>
                </ul>
              </div>
            </section>
            <section class="node-inspector-section">
              <h3 class="heading">Exploration / Index</h3>
              <ul class="node-details-list" id="selected-node-index-fields">
                <li class="empty">-</li>
              </ul>
            </section>
            <section class="node-inspector-section">
              <h3 class="heading">Raw Label</h3>
              <pre class="node-details-pre" id="selected-node-raw-label">-</pre>
            </section>
          </div>
        </section>
        <section class="card">
          <div class="detail-header">
            <h2 class="heading">Breakpoints</h2>
            <p class="subheading" id="pause-state">Status: unknown</p>
            <p class="subheading" id="last-breakpoint-hit">Last breakpoint hit: none</p>
          </div>
          <div class="detail-card detail-grid">
            <div class="controls">
              <select id="event-type-breakpoint">
                <option value="SearchIterationStarted">Iteration Started</option>
                <option value="NodeSelected">Node Selected</option>
                <option value="NodeOpeningPlanned">Node Opening Planned</option>
                <option value="ChildLinked">Child Linked</option>
                <option value="DirectValueAssigned">Direct Value Assigned</option>
                <option value="BackupStarted">Backup Started</option>
                <option value="BackupFinished">Backup Finished</option>
                <option value="SearchIterationCompleted">Iteration Completed</option>
              </select>
              <button id="add-event-breakpoint" type="button">Add Event Breakpoint</button>
              <input id="node-id-breakpoint" type="text" placeholder="Node id">
              <button id="add-node-breakpoint" type="button">Add Node Breakpoint</button>
              <select id="backup-flag-breakpoint">
                <option value="value_changed">Value Changed</option>
                <option value="pv_changed">PV Changed</option>
                <option value="over_changed">Over Changed</option>
              </select>
              <button id="add-backup-flag-breakpoint" type="button">Add Backup Breakpoint</button>
              <input id="iteration-breakpoint" type="number" min="0" step="1" placeholder="Iteration">
              <button id="add-iteration-breakpoint" type="button">Add Iteration Breakpoint</button>
              <button id="clear-breakpoints" type="button">Clear Breakpoints</button>
            </div>
            <ul class="breakpoint-list" id="breakpoint-list">
              <li class="empty">No breakpoints configured.</li>
            </ul>
          </div>
        </section>
      </main>
    </div>
    <script>
      const timelineElement = document.getElementById("timeline");
      const timelineMetaElement = document.getElementById("timeline-meta");
      const timelineSearchElement = document.getElementById("timeline-search");
      const timelineEventFilterElement = document.getElementById("timeline-event-filter");
      const jumpEntryIndexElement = document.getElementById("jump-entry-index");
      const jumpEntryButton = document.getElementById("jump-entry-button");
      const jumpNextBreakpointButton = document.getElementById("jump-next-breakpoint");
      const jumpNextPvChangeButton = document.getElementById("jump-next-pv-change");
      const jumpNextValueChangeButton = document.getElementById("jump-next-value-change");
      const jumpNextSelectedNodeEventButton = document.getElementById("jump-next-selected-node-event");
      const timelineFilterStatusElement = document.getElementById("timeline-filter-status");
      const previousButton = document.getElementById("previous-entry");
      const nextButton = document.getElementById("next-entry");
      const pauseButton = document.getElementById("pause-search");
      const resumeButton = document.getElementById("resume-search");
      const stepButton = document.getElementById("step-search");
      const addEventBreakpointButton = document.getElementById("add-event-breakpoint");
      const addNodeBreakpointButton = document.getElementById("add-node-breakpoint");
      const addBackupFlagBreakpointButton = document.getElementById("add-backup-flag-breakpoint");
      const addIterationBreakpointButton = document.getElementById("add-iteration-breakpoint");
      const clearBreakpointsButton = document.getElementById("clear-breakpoints");
      const autoFollowElement = document.getElementById("auto-follow-latest");
      const eventTypeBreakpointElement = document.getElementById("event-type-breakpoint");
      const nodeIdBreakpointElement = document.getElementById("node-id-breakpoint");
      const backupFlagBreakpointElement = document.getElementById("backup-flag-breakpoint");
      const iterationBreakpointElement = document.getElementById("iteration-breakpoint");
      const currentIndexElement = document.getElementById("current-index");
      const commandStatusElement = document.getElementById("command-status");
      const pauseStateElement = document.getElementById("pause-state");
      const lastBreakpointHitElement = document.getElementById("last-breakpoint-hit");
      const breakpointListElement = document.getElementById("breakpoint-list");
      const entryTitleElement = document.getElementById("entry-title");
      const entryTypeElement = document.getElementById("entry-type");
      const entrySummaryElement = document.getElementById("entry-summary");
      const snapshotStatusElement = document.getElementById("snapshot-status");
      const snapshotViewElement = document.getElementById("snapshot-view");
      const nodeSelectionStatusElement = document.getElementById("node-selection-status");
      const nodeListElement = document.getElementById("node-list");
      const nodeDetailsElement = document.getElementById("node-details");
      const selectedNodeIdElement = document.getElementById("selected-node-id");
      const selectedNodeDepthElement = document.getElementById("selected-node-depth");
      const selectedNodeRootElement = document.getElementById("selected-node-root");
      const selectedNodeStateTagElement = document.getElementById("selected-node-state-tag");
      const selectedNodeDirectValueElement = document.getElementById("selected-node-direct-value");
      const selectedNodeBackedUpValueElement = document.getElementById("selected-node-backed-up-value");
      const selectedNodePvElement = document.getElementById("selected-node-pv");
      const selectedNodeOverEventElement = document.getElementById("selected-node-over-event");
      const selectedNodeParentsElement = document.getElementById("selected-node-parents");
      const selectedNodeChildrenElement = document.getElementById("selected-node-children");
      const selectedNodeOutgoingEdgesElement = document.getElementById("selected-node-outgoing-edges");
      const selectedNodeIndexFieldsElement = document.getElementById("selected-node-index-fields");
      const selectedNodeRawLabelElement = document.getElementById("selected-node-raw-label");

      let entries = [];
      let visibleEntries = [];
      let selectedIndex = 0;
      let pollHandle = null;
      let isRefreshing = false;
      let currentPayload = null;
      let currentControlState = null;
      let currentSnapshotMetadata = null;
      let currentSnapshotMetadataFile = null;
      let currentGraphNodeMap = new Map();
      let selectedNodeId = null;
      let breakpointSequence = 0;
      let searchQuery = "";
      let selectedEventTypeFilter = "all";
      let timelineStatusMessage = "";

      function formatIndex(index) {
        return String(index).padStart(4, "0");
      }

      function hasActiveTimelineFilters() {
        return searchQuery.trim() !== "" || selectedEventTypeFilter !== "all";
      }

      function entryMatchesSearch(entry, query) {
        const normalizedQuery = query.trim().toLowerCase();
        if (!normalizedQuery) {
          return true;
        }

        const eventFieldsText = entry.event_fields
          ? JSON.stringify(entry.event_fields).toLowerCase()
          : "";
        const breakpointHitText = typeof entry.breakpoint_hit === "string"
          ? entry.breakpoint_hit.toLowerCase()
          : "";
        const haystack = [
          String(entry.index),
          entry.event_type || "",
          entry.event_summary || "",
          breakpointHitText,
          eventFieldsText,
        ].join(" ").toLowerCase();
        return haystack.includes(normalizedQuery);
      }

      function entryMatchesEventTypeFilter(entry, eventType) {
        if (eventType === "all") {
          return true;
        }
        return entry.event_type === eventType;
      }

      function computeVisibleEntries() {
        return entries.filter((entry) => (
          entryMatchesSearch(entry, searchQuery)
          && entryMatchesEventTypeFilter(entry, selectedEventTypeFilter)
        ));
      }

      function isSelectedEntryVisible() {
        return visibleEntries.some((entry) => entry.index === selectedIndex);
      }

      function currentVisibleEntryPosition() {
        return visibleEntries.findIndex((entry) => entry.index === selectedIndex);
      }

      function renderTimelineFilterStatus() {
        const parts = [`Showing ${visibleEntries.length} / ${entries.length} entries`];
        if (selectedEventTypeFilter !== "all") {
          parts.push(`Filter: ${selectedEventTypeFilter}`);
        }
        if (searchQuery.trim() !== "") {
          parts.push(`Search: "${searchQuery}"`);
        }
        if (timelineStatusMessage) {
          parts.push(timelineStatusMessage);
        }
        timelineFilterStatusElement.textContent = parts.join(" | ");
      }

      function setTimelineStatusMessage(message) {
        timelineStatusMessage = message;
        renderTimelineFilterStatus();
      }

      function clearTimelineStatusMessage() {
        if (!timelineStatusMessage) {
          return;
        }
        timelineStatusMessage = "";
        renderTimelineFilterStatus();
      }

      function entryHasBreakpointHit(entry) {
        return typeof entry.breakpoint_hit === "string" && entry.breakpoint_hit.length > 0;
      }

      function entryInvolvesNodeId(entry, nodeId) {
        if (!entry || !entry.event_fields || !nodeId) {
          return false;
        }

        const eventFields = entry.event_fields;
        return eventFields.node_id === nodeId
          || eventFields.parent_id === nodeId
          || eventFields.child_id === nodeId;
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

      function firstLabelLine(label) {
        if (typeof label !== "string" || label.length === 0) {
          return "(no label)";
        }
        return label.split("\n")[0];
      }

      function findNodeById(snapshotMetadata, nodeId) {
        if (!snapshotMetadata || !Array.isArray(snapshotMetadata.nodes) || !nodeId) {
          return null;
        }
        return snapshotMetadata.nodes.find((node) => node.node_id === nodeId) || null;
      }

      function findOutgoingEdges(snapshotMetadata, nodeId) {
        if (!snapshotMetadata || !Array.isArray(snapshotMetadata.edges) || !nodeId) {
          return [];
        }
        return snapshotMetadata.edges.filter((edge) => edge.parent_id === nodeId);
      }

      function nodeExists(snapshotMetadata, nodeId) {
        return findNodeById(snapshotMetadata, nodeId) !== null;
      }

      function extractNodeIdFromGraphvizGroup(nodeGroup) {
        const titleElement = Array.from(nodeGroup.children).find(
          (child) => child.tagName && child.tagName.toLowerCase() === "title"
        ) || nodeGroup.querySelector("title");
        if (!titleElement || !titleElement.textContent) {
          return null;
        }

        const candidateNodeId = titleElement.textContent.trim();
        if (!candidateNodeId) {
          return null;
        }
        if (!nodeExists(currentSnapshotMetadata, candidateNodeId)) {
          return null;
        }
        return candidateNodeId;
      }

      function renderGraphSelection() {
        currentGraphNodeMap.forEach((nodeGroup) => {
          nodeGroup.classList.remove("is-selected");
        });

        if (!selectedNodeId || !currentGraphNodeMap.has(selectedNodeId)) {
          return;
        }

        currentGraphNodeMap.get(selectedNodeId).classList.add("is-selected");
      }

      function selectNode(nodeId) {
        if (!nodeExists(currentSnapshotMetadata, nodeId)) {
          return;
        }
        selectedNodeId = nodeId;
        renderNodeList();
        renderNodeDetails();
        renderGraphSelection();
        updateControls();
      }

      function initializeGraphInteraction(svgElement) {
        currentGraphNodeMap = new Map();

        const nodeGroups = svgElement.querySelectorAll("g.node");
        nodeGroups.forEach((nodeGroup) => {
          const nodeId = extractNodeIdFromGraphvizGroup(nodeGroup);
          if (!nodeId) {
            return;
          }

          currentGraphNodeMap.set(nodeId, nodeGroup);
          nodeGroup.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            selectNode(nodeId);
          });
        });

        renderGraphSelection();
      }

      async function loadInlineSvg(snapshotFile) {
        const response = await fetch(snapshotFile, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return await response.text();
      }

      function renderInlineSvg(svgText) {
        snapshotViewElement.replaceChildren();
        snapshotViewElement.innerHTML = svgText;
        return snapshotViewElement.querySelector("svg");
      }

      function renderNodeList() {
        nodeListElement.replaceChildren();

        if (!currentSnapshotMetadata || !Array.isArray(currentSnapshotMetadata.nodes)) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "No snapshot nodes available yet.";
          nodeListElement.appendChild(emptyItem);
          return;
        }

        if (currentSnapshotMetadata.nodes.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "Snapshot contains no nodes.";
          nodeListElement.appendChild(emptyItem);
          return;
        }

        currentSnapshotMetadata.nodes.forEach((node) => {
          const item = document.createElement("li");
          const button = document.createElement("button");
          button.type = "button";
          button.className = "node-list-item";
          if (node.node_id === selectedNodeId) {
            button.classList.add("is-selected");
          }

          const label = document.createElement("span");
          label.className = "node-list-label";
          label.textContent = `Node ${node.node_id}`;

          const summary = document.createElement("span");
          summary.className = "node-list-summary";
          summary.textContent = firstLabelLine(node.label);

          button.appendChild(label);
          button.appendChild(summary);
          button.addEventListener("click", () => {
            selectNode(node.node_id);
          });
          item.appendChild(button);
          nodeListElement.appendChild(item);
        });
      }

      function renderListItems(container, items, emptyLabel) {
        container.replaceChildren();

        if (!Array.isArray(items) || items.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = emptyLabel;
          container.appendChild(emptyItem);
          return;
        }

        items.forEach((itemText) => {
          const item = document.createElement("li");
          item.textContent = itemText;
          container.appendChild(item);
        });
      }

      function renderNodeInspector(node) {
        const childIds = Array.isArray(node.child_ids) ? node.child_ids : [];
        const parents = Array.isArray(node.parent_ids) ? node.parent_ids : [];
        const edgeLabelsByChild = node.edge_labels_by_child && typeof node.edge_labels_by_child === "object"
          ? node.edge_labels_by_child
          : {};
        const indexFields = node.index_fields && typeof node.index_fields === "object"
          ? Object.entries(node.index_fields).map(([key, value]) => `${key} = ${value}`)
          : [];
        const outgoingEdges = childIds.length > 0
          ? childIds.map((childId) => (
            edgeLabelsByChild[childId] ? `${edgeLabelsByChild[childId]} -> ${childId}` : childId
          ))
          : findOutgoingEdges(currentSnapshotMetadata, node.node_id).map((edge) => (
            edge.label ? `${edge.label} -> ${edge.child_id}` : edge.child_id
          ));

        selectedNodeIdElement.textContent = node.node_id || "-";
        selectedNodeDepthElement.textContent = String(node.depth ?? "-");
        selectedNodeRootElement.textContent = node.node_id === currentSnapshotMetadata.root_id
          ? "yes"
          : "no";
        selectedNodeStateTagElement.textContent = node.state_tag || "-";
        selectedNodeDirectValueElement.textContent = node.direct_value || "-";
        selectedNodeBackedUpValueElement.textContent = node.backed_up_value || "-";
        selectedNodePvElement.textContent = node.principal_variation || "-";
        selectedNodeOverEventElement.textContent = node.over_event || "-";
        selectedNodeParentsElement.textContent = parents.length > 0 ? parents.join(", ") : "-";
        selectedNodeChildrenElement.textContent = childIds.length > 0 ? childIds.join(", ") : "-";
        renderListItems(selectedNodeOutgoingEdgesElement, outgoingEdges, "none");
        renderListItems(selectedNodeIndexFieldsElement, indexFields, "none");
        selectedNodeRawLabelElement.textContent = typeof node.label === "string"
          ? node.label
          : "-";
      }

      function renderNodeDetails() {
        if (!currentSnapshotMetadata) {
          nodeSelectionStatusElement.textContent = "No node selected";
          selectedNodeIdElement.textContent = "-";
          selectedNodeDepthElement.textContent = "-";
          selectedNodeRootElement.textContent = "-";
          selectedNodeStateTagElement.textContent = "-";
          selectedNodeDirectValueElement.textContent = "-";
          selectedNodeBackedUpValueElement.textContent = "-";
          selectedNodePvElement.textContent = "-";
          selectedNodeOverEventElement.textContent = "-";
          selectedNodeParentsElement.textContent = "-";
          selectedNodeChildrenElement.textContent = "-";
          renderListItems(selectedNodeOutgoingEdgesElement, [], "No snapshot metadata available yet.");
          renderListItems(selectedNodeIndexFieldsElement, [], "No snapshot metadata available yet.");
          selectedNodeRawLabelElement.textContent = "-";
          return;
        }

        if (!selectedNodeId) {
          nodeSelectionStatusElement.textContent = "No node selected";
          selectedNodeIdElement.textContent = "-";
          selectedNodeDepthElement.textContent = "-";
          selectedNodeRootElement.textContent = "-";
          selectedNodeStateTagElement.textContent = "-";
          selectedNodeDirectValueElement.textContent = "-";
          selectedNodeBackedUpValueElement.textContent = "-";
          selectedNodePvElement.textContent = "-";
          selectedNodeOverEventElement.textContent = "-";
          selectedNodeParentsElement.textContent = "-";
          selectedNodeChildrenElement.textContent = "-";
          renderListItems(selectedNodeOutgoingEdgesElement, [], "Select a node to inspect outgoing edges.");
          renderListItems(selectedNodeIndexFieldsElement, [], "Select a node to inspect index fields.");
          selectedNodeRawLabelElement.textContent = "-";
          return;
        }

        const node = findNodeById(currentSnapshotMetadata, selectedNodeId);
        if (!node) {
          nodeSelectionStatusElement.textContent = "No node selected";
          selectedNodeIdElement.textContent = "-";
          selectedNodeDepthElement.textContent = "-";
          selectedNodeRootElement.textContent = "-";
          selectedNodeStateTagElement.textContent = "-";
          selectedNodeDirectValueElement.textContent = "-";
          selectedNodeBackedUpValueElement.textContent = "-";
          selectedNodePvElement.textContent = "-";
          selectedNodeOverEventElement.textContent = "-";
          selectedNodeParentsElement.textContent = "-";
          selectedNodeChildrenElement.textContent = "-";
          renderListItems(selectedNodeOutgoingEdgesElement, [], "Selected node is not present in this snapshot.");
          renderListItems(selectedNodeIndexFieldsElement, [], "Selected node is not present in this snapshot.");
          selectedNodeRawLabelElement.textContent = "-";
          return;
        }

        nodeSelectionStatusElement.textContent = `Selected node: ${selectedNodeId}`;
        renderNodeInspector(node);
      }

      function renderTimeline() {
        timelineElement.replaceChildren();

        if (entries.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "No replay entries available.";
          timelineElement.appendChild(emptyItem);
          renderTimelineFilterStatus();
          return;
        }

        if (visibleEntries.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "No timeline entries match the current search/filter.";
          timelineElement.appendChild(emptyItem);
          renderTimelineFilterStatus();
          return;
        }

        visibleEntries.forEach((entry) => {
          const listItem = document.createElement("li");
          const button = document.createElement("button");
          button.type = "button";
          button.className = "timeline-item";
          if (entry.index === selectedIndex) {
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
            selectEntry(entry.index);
          });

          listItem.appendChild(button);
          timelineElement.appendChild(listItem);
        });

        renderTimelineFilterStatus();
      }

      function updateControls() {
        const hasEntries = entries.length > 0;
        const canControl = Boolean(
          currentPayload && currentPayload.is_live && !currentPayload.is_complete
        );
        const hasSelectedNode = Boolean(selectedNodeId);
        const visiblePosition = currentVisibleEntryPosition();
        if (hasActiveTimelineFilters()) {
          previousButton.disabled = visiblePosition <= 0;
          nextButton.disabled = (
            visiblePosition < 0 || visiblePosition >= visibleEntries.length - 1
          );
        } else {
          previousButton.disabled = !hasEntries || selectedIndex <= 0;
          nextButton.disabled = !hasEntries || selectedIndex >= entries.length - 1;
        }
        pauseButton.disabled = !canControl;
        resumeButton.disabled = !canControl;
        stepButton.disabled = !canControl;
        addEventBreakpointButton.disabled = !canControl;
        addNodeBreakpointButton.disabled = !canControl;
        addBackupFlagBreakpointButton.disabled = !canControl;
        addIterationBreakpointButton.disabled = !canControl;
        clearBreakpointsButton.disabled = !canControl;
        jumpEntryButton.disabled = !hasEntries;
        jumpNextBreakpointButton.disabled = !hasEntries;
        jumpNextPvChangeButton.disabled = !hasEntries;
        jumpNextValueChangeButton.disabled = !hasEntries;
        jumpNextSelectedNodeEventButton.disabled = !hasEntries || !hasSelectedNode;
      }

      function renderCommandStatus(payload) {
        if (!payload || !payload.is_live) {
          commandStatusElement.textContent = "Static replay: live control unavailable";
          return;
        }

        if (payload.is_complete) {
          commandStatusElement.textContent = "Session complete";
          return;
        }

        commandStatusElement.textContent = "Live control enabled";
      }

      function renderBreakpointList(breakpoints) {
        breakpointListElement.replaceChildren();

        if (!Array.isArray(breakpoints) || breakpoints.length === 0) {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty";
          emptyItem.textContent = "No breakpoints configured.";
          breakpointListElement.appendChild(emptyItem);
          return;
        }

        breakpoints.forEach((breakpoint) => {
          const item = document.createElement("li");
          item.className = "breakpoint-item";
          item.textContent = formatBreakpoint(breakpoint);
          breakpointListElement.appendChild(item);
        });
      }

      function formatBreakpoint(breakpoint) {
        switch (breakpoint.kind) {
          case "event_type":
            return `${breakpoint.id} | event type = ${breakpoint.event_type}`;
          case "node_id":
            return `${breakpoint.id} | node id = ${breakpoint.node_id}`;
          case "backup_flag":
            return `${breakpoint.id} | backup flag = ${breakpoint.flag_name}`;
          case "iteration":
            return `${breakpoint.id} | iteration = ${breakpoint.iteration_index}`;
          default:
            return breakpoint.id || "breakpoint";
        }
      }

      function renderControlState(controlState) {
        currentControlState = controlState;

        if (!controlState) {
          pauseStateElement.textContent = "Status: unknown";
          lastBreakpointHitElement.textContent = "Last breakpoint hit: none";
          renderBreakpointList([]);
          updateControls();
          return;
        }

        pauseStateElement.textContent = controlState.paused
          ? "Status: paused"
          : "Status: running";
        lastBreakpointHitElement.textContent = controlState.last_breakpoint_hit
          ? `Last breakpoint hit: ${controlState.last_breakpoint_hit}`
          : "Last breakpoint hit: none";
        renderBreakpointList(controlState.breakpoints);
        updateControls();
      }

      function renderEmptyDetails(message) {
        currentIndexElement.textContent = "No entry selected";
        entryTitleElement.textContent = "Replay details";
        entryTypeElement.textContent = "-";
        entrySummaryElement.textContent = message;
        snapshotStatusElement.textContent = "No snapshot loaded";
        currentSnapshotMetadata = null;
        currentSnapshotMetadataFile = null;
        currentGraphNodeMap = new Map();
        selectedNodeId = null;
        snapshotViewElement.replaceChildren();
        const emptyMessage = document.createElement("div");
        emptyMessage.className = "empty";
        emptyMessage.textContent = message;
        snapshotViewElement.appendChild(emptyMessage);
        renderNodeList();
        renderNodeDetails();
        updateControls();
      }

      function startPolling() {
        if (pollHandle !== null) {
          return;
        }

        pollHandle = window.setInterval(() => {
          void refreshTrace();
        }, 1000);
      }

      function stopPolling() {
        if (pollHandle === null) {
          return;
        }

        window.clearInterval(pollHandle);
        pollHandle = null;
      }

      function syncPolling(payload) {
        if (payload && payload.is_live && !payload.is_complete) {
          startPolling();
          return;
        }

        stopPolling();
      }

      function renderTimelineMeta(payload) {
        const entryCount = payload && Number.isInteger(payload.entry_count)
          ? payload.entry_count
          : entries.length;
        const mode = payload && payload.is_live
          ? payload.is_complete
            ? "live session complete"
            : "live session updating"
          : "static replay";
        timelineMetaElement.textContent = `${entryCount} entries loaded | ${mode}`;
      }

      function synchronizeSelectionWithVisibleEntries() {
        if (entries.length === 0) {
          selectedIndex = 0;
          return;
        }

        if (selectedIndex >= entries.length) {
          selectedIndex = entries.length - 1;
        }

        if (visibleEntries.length === 0) {
          return;
        }

        if (!isSelectedEntryVisible()) {
          selectedIndex = visibleEntries[0].index;
        }
      }

      function applyTimelineFilters() {
        visibleEntries = computeVisibleEntries();
        synchronizeSelectionWithVisibleEntries();
        renderTimeline();
        updateControls();
      }

      async function refreshTimelineViewFromControls() {
        applyTimelineFilters();
        if (entries.length === 0) {
          renderEmptyDetails("No replay entries available.");
          return;
        }
        if (visibleEntries.length === 0 && hasActiveTimelineFilters()) {
          renderEmptyDetails("No timeline entries match the current search/filter.");
          return;
        }
        await renderDetails();
      }

      async function sendCommand(commandName) {
        try {
          const response = await fetch("/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: commandName }),
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          commandStatusElement.textContent = `Sent ${commandName} command`;
          await refreshTrace();
        } catch (error) {
          commandStatusElement.textContent = `Command failed: ${error}`;
        }
      }

      function nextBreakpointId(prefix) {
        breakpointSequence += 1;
        return `${prefix}-${Date.now()}-${breakpointSequence}`;
      }

      async function addBreakpoint(payload) {
        try {
          const response = await fetch("/breakpoints", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "add",
              breakpoint: payload,
            }),
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          commandStatusElement.textContent = `Added breakpoint ${payload.id}`;
          await refreshTrace();
        } catch (error) {
          commandStatusElement.textContent = `Breakpoint failed: ${error}`;
        }
      }

      async function clearBreakpoints() {
        try {
          const response = await fetch("/breakpoints", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "clear" }),
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          commandStatusElement.textContent = "Cleared breakpoints";
          await refreshTrace();
        } catch (error) {
          commandStatusElement.textContent = `Breakpoint clear failed: ${error}`;
        }
      }

      async function jumpToEntryIndex(index) {
        if (!Number.isInteger(index)) {
          setTimelineStatusMessage("Enter a valid entry index.");
          return;
        }

        const targetEntry = entries.find((entry) => entry.index === index);
        if (!targetEntry) {
          setTimelineStatusMessage(`Entry index ${index} was not found.`);
          return;
        }

        clearTimelineStatusMessage();
        await selectEntry(targetEntry.index);
      }

      async function jumpToNextMatching(predicate, missingMessage) {
        for (let index = selectedIndex + 1; index < entries.length; index += 1) {
          const entry = entries[index];
          if (predicate(entry)) {
            clearTimelineStatusMessage();
            await selectEntry(index);
            return;
          }
        }

        setTimelineStatusMessage(missingMessage);
      }

      async function jumpToNextBreakpointHit() {
        await jumpToNextMatching(
          (entry) => entryHasBreakpointHit(entry),
          "No later breakpoint hit found."
        );
      }

      async function jumpToNextPvChange() {
        await jumpToNextMatching(
          (entry) => entry.event_fields && entry.event_fields.pv_changed === true,
          "No later PV change found."
        );
      }

      async function jumpToNextValueChange() {
        await jumpToNextMatching(
          (entry) => entry.event_fields && entry.event_fields.value_changed === true,
          "No later value change found."
        );
      }

      async function jumpToNextSelectedNodeEvent() {
        if (!selectedNodeId) {
          setTimelineStatusMessage("Select a node before jumping to node events.");
          return;
        }

        await jumpToNextMatching(
          (entry) => entryInvolvesNodeId(entry, selectedNodeId),
          `No later event found for node ${selectedNodeId}.`
        );
      }

      async function loadSnapshotMetadataForEntry(entryIndex) {
        const snapshotEntry = nearestSnapshotAtOrBefore(entryIndex);
        if (!snapshotEntry || !snapshotEntry.snapshot_metadata_file) {
          currentSnapshotMetadata = null;
          currentSnapshotMetadataFile = null;
          selectedNodeId = null;
          renderNodeList();
          renderNodeDetails();
          renderGraphSelection();
          return;
        }

        if (currentSnapshotMetadataFile !== snapshotEntry.snapshot_metadata_file) {
          const response = await fetch(snapshotEntry.snapshot_metadata_file, {
            cache: "no-store",
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          currentSnapshotMetadata = await response.json();
          currentSnapshotMetadataFile = snapshotEntry.snapshot_metadata_file;
        }

        if (!findNodeById(currentSnapshotMetadata, selectedNodeId)) {
          selectedNodeId = null;
        }

        renderNodeList();
        renderNodeDetails();
        renderGraphSelection();
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
          currentSnapshotMetadata = null;
          currentSnapshotMetadataFile = null;
          currentGraphNodeMap = new Map();
          selectedNodeId = null;
          renderNodeList();
          renderNodeDetails();
          return;
        }

        const sourceLabel = snapshotEntry.index === entryIndex
          ? `Snapshot attached to entry [${formatIndex(snapshotEntry.index)}]`
          : `Using nearest prior snapshot from entry [${formatIndex(snapshotEntry.index)}]`;
        snapshotStatusElement.textContent = sourceLabel;

        if (snapshotEntry.snapshot_file.endsWith(".dot")) {
          currentGraphNodeMap = new Map();
          const response = await fetch(snapshotEntry.snapshot_file, { cache: "no-store" });
          const dotSource = await response.text();
          const block = document.createElement("pre");
          block.textContent = dotSource;
          snapshotViewElement.appendChild(block);
          renderGraphSelection();
          return;
        }

        const svgText = await loadInlineSvg(snapshotEntry.snapshot_file);
        const svgElement = renderInlineSvg(svgText);
        if (!svgElement) {
          currentGraphNodeMap = new Map();
          const emptyMessage = document.createElement("div");
          emptyMessage.className = "empty";
          emptyMessage.textContent = "Unable to render inline SVG snapshot.";
          snapshotViewElement.replaceChildren();
          snapshotViewElement.appendChild(emptyMessage);
          return;
        }
        initializeGraphInteraction(svgElement);
      }

      async function renderDetails() {
        if (entries.length === 0) {
          renderEmptyDetails("No replay entries available.");
          return;
        }

        if (visibleEntries.length === 0 && hasActiveTimelineFilters()) {
          renderEmptyDetails("No timeline entries match the current search/filter.");
          return;
        }

        const entry = entries[selectedIndex];
        currentIndexElement.textContent = `Entry ${selectedIndex + 1} of ${entries.length}`;
        entryTitleElement.textContent = `[${formatIndex(entry.index)}] ${entry.event_summary}`;
        entryTypeElement.textContent = entry.event_type;
        entrySummaryElement.textContent = entry.event_summary;
        updateControls();
        // Metadata must be available before inline SVG interaction is initialized,
        // because Graphviz node ids are validated against the structured snapshot.
        await loadSnapshotMetadataForEntry(selectedIndex);
        await renderSnapshot(selectedIndex);
      }

      async function selectEntry(index) {
        if (!Number.isInteger(index) || index < 0 || index >= entries.length) {
          return;
        }
        selectedIndex = index;
        renderTimeline();
        await renderDetails();
      }

      async function fetchSessionPayload() {
        const sessionResponse = await fetch("session.json", { cache: "no-store" });
        if (sessionResponse.ok) {
          return await sessionResponse.json();
        }
        if (sessionResponse.status !== 404) {
          throw new Error(`HTTP ${sessionResponse.status}`);
        }

        const traceResponse = await fetch("trace.json", { cache: "no-store" });
        if (!traceResponse.ok) {
          throw new Error(`HTTP ${traceResponse.status}`);
        }
        return await traceResponse.json();
      }

      async function fetchControlState() {
        const response = await fetch("control_state.json", { cache: "no-store" });
        if (response.ok) {
          return await response.json();
        }
        if (response.status === 404) {
          return null;
        }
        throw new Error(`HTTP ${response.status}`);
      }

      async function refreshTrace() {
        if (isRefreshing) {
          return;
        }

        isRefreshing = true;
        try {
          const payload = await fetchSessionPayload();
          const controlState = await fetchControlState();
          currentPayload = payload;
          entries = Array.isArray(payload.entries) ? payload.entries : [];

          if (entries.length > 0) {
            if (autoFollowElement.checked && !hasActiveTimelineFilters()) {
              selectedIndex = entries.length - 1;
            } else if (selectedIndex >= entries.length) {
              selectedIndex = entries.length - 1;
            }
          } else {
            selectedIndex = 0;
          }

          renderTimelineMeta(payload);
          renderCommandStatus(payload);
          renderControlState(controlState);
          applyTimelineFilters();

          if (entries.length > 0 && (visibleEntries.length > 0 || !hasActiveTimelineFilters())) {
            await renderDetails();
          } else {
            renderEmptyDetails("No timeline entries match the current search/filter.");
          }

          syncPolling(payload);
        } catch (error) {
          stopPolling();
          currentPayload = null;
          currentControlState = null;
          timelineMetaElement.textContent = "Failed to load replay payload";
          commandStatusElement.textContent = "Live control unavailable";
          pauseStateElement.textContent = "Status: unknown";
          lastBreakpointHitElement.textContent = "Last breakpoint hit: none";
          renderEmptyDetails(`Unable to load replay payload: ${error}`);
        } finally {
          isRefreshing = false;
        }
      }

      previousButton.addEventListener("click", async () => {
        clearTimelineStatusMessage();
        if (hasActiveTimelineFilters()) {
          const visiblePosition = currentVisibleEntryPosition();
          if (visiblePosition > 0) {
            await selectEntry(visibleEntries[visiblePosition - 1].index);
          }
          return;
        }

        if (selectedIndex > 0) {
          await selectEntry(selectedIndex - 1);
        }
      });

      nextButton.addEventListener("click", async () => {
        clearTimelineStatusMessage();
        if (hasActiveTimelineFilters()) {
          const visiblePosition = currentVisibleEntryPosition();
          if (visiblePosition >= 0 && visiblePosition < visibleEntries.length - 1) {
            await selectEntry(visibleEntries[visiblePosition + 1].index);
          }
          return;
        }

        if (selectedIndex < entries.length - 1) {
          await selectEntry(selectedIndex + 1);
        }
      });

      timelineSearchElement.addEventListener("input", async () => {
        searchQuery = timelineSearchElement.value.trim();
        clearTimelineStatusMessage();
        await refreshTimelineViewFromControls();
      });

      timelineEventFilterElement.addEventListener("change", async () => {
        selectedEventTypeFilter = timelineEventFilterElement.value;
        clearTimelineStatusMessage();
        await refreshTimelineViewFromControls();
      });

      jumpEntryButton.addEventListener("click", async () => {
        const requestedIndex = Number.parseInt(jumpEntryIndexElement.value, 10);
        await jumpToEntryIndex(requestedIndex);
      });

      jumpEntryIndexElement.addEventListener("keydown", async (event) => {
        if (event.key !== "Enter") {
          return;
        }
        const requestedIndex = Number.parseInt(jumpEntryIndexElement.value, 10);
        await jumpToEntryIndex(requestedIndex);
      });

      jumpNextBreakpointButton.addEventListener("click", async () => {
        await jumpToNextBreakpointHit();
      });

      jumpNextPvChangeButton.addEventListener("click", async () => {
        await jumpToNextPvChange();
      });

      jumpNextValueChangeButton.addEventListener("click", async () => {
        await jumpToNextValueChange();
      });

      jumpNextSelectedNodeEventButton.addEventListener("click", async () => {
        await jumpToNextSelectedNodeEvent();
      });

      autoFollowElement.addEventListener("change", async () => {
        if (autoFollowElement.checked && entries.length > 0 && !hasActiveTimelineFilters()) {
          clearTimelineStatusMessage();
          await selectEntry(entries.length - 1);
        }
      });

      pauseButton.addEventListener("click", async () => {
        await sendCommand("pause");
      });

      resumeButton.addEventListener("click", async () => {
        await sendCommand("resume");
      });

      stepButton.addEventListener("click", async () => {
        await sendCommand("step");
      });

      addEventBreakpointButton.addEventListener("click", async () => {
        await addBreakpoint({
          kind: "event_type",
          id: nextBreakpointId("bp-event"),
          enabled: true,
          event_type: eventTypeBreakpointElement.value,
        });
      });

      addNodeBreakpointButton.addEventListener("click", async () => {
        if (!nodeIdBreakpointElement.value) {
          commandStatusElement.textContent = "Node breakpoint requires a node id";
          return;
        }

        await addBreakpoint({
          kind: "node_id",
          id: nextBreakpointId("bp-node"),
          enabled: true,
          node_id: nodeIdBreakpointElement.value,
        });
      });

      addBackupFlagBreakpointButton.addEventListener("click", async () => {
        await addBreakpoint({
          kind: "backup_flag",
          id: nextBreakpointId("bp-backup"),
          enabled: true,
          flag_name: backupFlagBreakpointElement.value,
        });
      });

      addIterationBreakpointButton.addEventListener("click", async () => {
        if (!iterationBreakpointElement.value) {
          commandStatusElement.textContent = "Iteration breakpoint requires a number";
          return;
        }

        await addBreakpoint({
          kind: "iteration",
          id: nextBreakpointId("bp-iteration"),
          enabled: true,
          iteration_index: Number.parseInt(iterationBreakpointElement.value, 10),
        });
      });

      clearBreakpointsButton.addEventListener("click", async () => {
        await clearBreakpoints();
      });

      void refreshTrace();
    </script>
  </body>
</html>
"""


__all__ = ["render_replay_index_html"]
