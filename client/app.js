const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const serversEl = document.getElementById("servers");

function appendMessage(role, text, toolCalls = []) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;

  if (toolCalls.length) {
    const trace = document.createElement("div");
    trace.className = "tool-trace";
    trace.innerHTML = "<strong>Tool trace</strong>";
    for (const step of toolCalls) {
      const item = document.createElement("div");
      item.className = "tool-step";
      item.innerHTML = `
        <div><code>${step.tool}</code></div>
        <div>Args: ${JSON.stringify(step.arguments)}</div>
        <div>Result: ${step.result.slice(0, 400)}${step.result.length > 400 ? "..." : ""}</div>
      `;
      trace.appendChild(item);
    }
    node.appendChild(trace);
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function loadPlatform() {
  try {
    const response = await fetch("/platform");
    const data = await response.json();
    const server = data.server;
    serversEl.innerHTML = "";
    const card = document.createElement("div");
    card.className = "server-card";
    card.innerHTML = `
      <strong>${server.name}</strong>
      <small>${server.description || server.server_id}</small>
      <div><small>${server.tools || "No tools"}</small></div>
    `;
    serversEl.appendChild(card);
  } catch (error) {
    serversEl.textContent = "Could not load platform info.";
  }
}

async function sendMessage(text) {
  const query = text.trim();
  if (!query) return;

  appendMessage("user", query);
  input.value = "";
  sendBtn.disabled = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Request failed");
    }

    const data = await response.json();
    appendMessage("assistant", data.answer, data.tool_calls);
  } catch (error) {
    appendMessage("system", `Error: ${error.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    const query = button.getAttribute("data-query");
    input.value = query;
    sendMessage(query);
  });
});

loadPlatform();
appendMessage("system", "Connected. Ask a question and the agent will route to MCP tools automatically.");
