const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const levelSelect = document.querySelector("#style-level");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");
const errorMessage = document.querySelector("#error-message");

let sessionId = null;
let busy = false;

function addMessage(text, role) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "bot-message"}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "T";
    article.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  if (role === "assistant") {
    const speaker = document.createElement("span");
    speaker.className = "speaker";
    speaker.textContent = "The assistant";
    content.append(speaker);
  }
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  content.append(paragraph);
  article.append(content);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;

  errorMessage.hidden = true;
  addMessage(message, "user");
  input.value = "";
  input.style.height = "auto";
  busy = true;
  sendButton.disabled = true;
  const pending = addMessage("Thinking through the answer (and its eventual detours)...", "assistant");
  pending.classList.add("typing");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, level: levelSelect.value, sessionId }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The reply could not be generated.");
    sessionId = result.sessionId;
    pending.remove();
    addMessage(result.reply, "assistant");
    if (result.evaluation) {
      const status = result.evaluation.fallback
        ? " · used clear original"
        : result.evaluation.adjusted ? " · automatically simplified" : "";
      addMessage(
        `Meaning fidelity: ${result.evaluation.fidelity}% · Clarity: ${result.evaluation.clarity}%${status}`,
        "meta",
      );
    }
  } catch (error) {
    pending.remove();
    errorMessage.textContent = error.message || "The server could not be reached. Please try again.";
    errorMessage.hidden = false;
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

clearButton.addEventListener("click", async () => {
  if (busy) return;
  errorMessage.hidden = true;
  try {
    const response = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });
    if (!response.ok) throw new Error("The conversation could not be reset.");
    sessionId = null;
    messages.replaceChildren();
    addMessage(
      "Ask me something, and I shall answer plainly first—which is to say, before making a graceful detour.",
      "assistant",
    );
    input.focus();
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  }
});
