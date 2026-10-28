(() => {
  const app = document.getElementById("app");
  const conversation = document.getElementById("conversation");
  const input = document.getElementById("questionInput");
  const sendButton = document.getElementById("sendButton");
  const config = window.CHAT_CONFIG || { mode: "demo", endpoint: "/api/chat" };
  let messages = [];
  let conversationId = null;
  let showingExample = false;
  let viewVersion = 0;
  let pendingChat = null;
  let resetReady = true;

  function setWelcome(value) {
    app.classList.toggle("welcome-mode", value);
    document.getElementById("topic").textContent = value ? "New chat" : (showingExample ? "Cover and visibility" : "Conversation");
  }

  async function showExample() {
    viewVersion++;
    pendingChat?.abort();
    pendingChat = null;
    showingExample = true;
    messages = [];
    conversationId = null;
    // Restore the static example without sending anything to the backend.
    conversation.innerHTML = exampleMarkup;
    attachExampleCitations();
    setWelcome(false);
    await resetServerConversation();
  }

  function attachExampleCitations() {
    conversation.querySelectorAll(".cite").forEach(button => {
      button.addEventListener("click", () => {
        const details = conversation.querySelector("#inlineSources");
        if (!details) return;
        details.open = true;
        details.querySelectorAll(".source-item").forEach(item => item.classList.toggle("focused", item.id === button.dataset.source));
        details.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    });
  }

  async function newChat() {
    viewVersion++;
    pendingChat?.abort();
    pendingChat = null;
    messages = [];
    conversationId = null;
    showingExample = false;
    conversation.replaceChildren();
    input.value = "";
    setWelcome(true);
    await resetServerConversation();
  }

  async function resetServerConversation() {
    if (config.mode === "api") {
      resetReady = false;
      sendButton.disabled = true;
      try {
        const response = await fetch(config.resetEndpoint || "/api/chat/reset", {
          method: "POST",
          credentials: "same-origin"
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        resetReady = true;
      } catch {
        showNotice("Could not reset the server conversation. Please try again.");
      } finally {
        sendButton.disabled = !resetReady;
      }
    }
  }

  function showNotice(message) {
    const notice = document.getElementById("notice");
    notice.textContent = message;
    notice.classList.add("show");
    setTimeout(() => notice.classList.remove("show"), 5000);
  }

  function addUserMessage(text) {
    const element = document.createElement("div");
    element.className = "user-message";
    element.textContent = text;
    conversation.append(element);
    return element;
  }

  function addAnswer(answer, sources = []) {
    const section = document.createElement("section");
    section.className = "answer answer-message";
    const icon = document.createElement("div");
    icon.className = "answer-icon";
    icon.textContent = "✦";
    icon.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "answerbody";
    String(answer || "No answer available.").split(/\n\s*\n/).forEach(paragraph => {
      const p = document.createElement("p");
      p.textContent = paragraph;
      body.append(p);
    });
    if (sources.length) {
      const details = document.createElement("details");
      details.className = "sources-inline";
      const summary = document.createElement("summary");
      summary.textContent = `Show sources (${sources.length})`;
      details.append(summary);
      sources.forEach(source => {
        const item = document.createElement("div");
        item.className = "source-item";
        const title = document.createElement("strong");
        title.textContent = [source.title, source.locator].filter(Boolean).join(" · ") || "Source";
        item.append(title);
        if (source.excerpt) {
          const excerpt = document.createElement("div");
          excerpt.textContent = source.excerpt;
          item.append(excerpt);
        }
        if (source.url && /^https?:\/\//i.test(source.url)) {
          const link = document.createElement("a");
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = "Open source ↗";
          item.append(link);
        }
        details.append(item);
      });
      body.append(details);
    }
    section.append(icon, body);
    conversation.append(section);
    section.scrollIntoView({ block: "end", behavior: "smooth" });
  }

  async function getReply(question, signal) {
    if (config.mode === "demo") {
      return {
        answer: question.toLowerCase().includes("visib")
          ? "A model is visible if any part of it can be seen from the observing model. Check the terrain rules when deciding whether it can be targeted."
          : "This is a sample response. The complete version will use the project sources.",
        sources: question.toLowerCase().includes("visib")
          ? [{ title: "Core Rules", locator: "Visibility", excerpt: "Illustrative example: verify the page and passage in the final RAG." }]
          : []
      };
    }
    const response = await fetch(config.endpoint, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
      signal
    });
    const result = await response.json();
    if (!response.ok || result.status === "error") throw new Error(result.error || `HTTP ${response.status}`);
    if (typeof result.answer !== "string") throw new Error("Server response is missing the answer field.");
    return { answer: result.answer, sources: Array.isArray(result.sources) ? result.sources : [], conversation_id: result.conversation_id };
  }

  const exampleMarkup = conversation.innerHTML;
  const keyInput = document.getElementById("apiKey");
  const keyButton = document.getElementById("saveKey");
  const keyStatus = document.getElementById("keyStatus");
  if (config.mode === "api") {
    keyInput.disabled = false;
    keyButton.disabled = false;
    keyStatus.textContent = "The key will be used for this session.";
  }
  document.getElementById("keyForm").addEventListener("submit", async event => {
    event.preventDefault();
    if (config.mode !== "api") return;
    const key = keyInput.value.trim();
    if (!key) return;
    keyButton.disabled = true;
    keyStatus.classList.remove("error");
    keyStatus.textContent = "Setting key…";
    try {
      const response = await fetch(config.keyEndpoint || "/api/session/key", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      keyInput.value = "";
      app.classList.add("book-unlocked");
      document.getElementById("archiveBook").setAttribute("aria-label", "Open book, archive unlocked");
      keyStatus.textContent = "Key set for this session. The archive is open.";
    } catch (error) {
      keyStatus.classList.add("error");
      keyStatus.textContent = `Could not set the key (${error.message}).`;
    } finally {
      keyButton.disabled = false;
    }
  });
  attachExampleCitations();
  document.getElementById("newChat").addEventListener("click", newChat);
  document.getElementById("showExample").addEventListener("click", showExample);
  document.getElementById("exampleNav").addEventListener("click", showExample);
  document.getElementById("focusQuestion").addEventListener("click", () => input.focus());
  document.getElementById("composer").addEventListener("submit", async event => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || sendButton.disabled || !resetReady) return;
    if (showingExample || app.classList.contains("welcome-mode")) conversation.replaceChildren();
    showingExample = false;
    messages.push({ role: "user", content: question });
    setWelcome(false);
    const currentView = viewVersion;
    const userBubble = addUserMessage(question);
    input.value = "";
    sendButton.disabled = true;
    const controller = new AbortController();
    pendingChat = controller;
    const loading = document.createElement("div");
    loading.className = "loading";
    loading.textContent = "Checking the rules…";
    conversation.append(loading);
    try {
      const reply = await getReply(question, controller.signal);
      if (currentView !== viewVersion) return;
      loading.remove();
      conversationId = reply.conversation_id ?? conversationId;
      messages.push({ role: "assistant", content: reply.answer });
      addAnswer(reply.answer, reply.sources);
    } catch (error) {
      if (currentView !== viewVersion) return;
      if (error.name === "AbortError") return;
      loading.remove();
      messages.pop();
      userBubble.remove();
      input.value = question;
      const message = document.createElement("div");
      message.className = "error-message";
      message.textContent = `Could not get a response. ${error.message}`;
      conversation.append(message);
    } finally {
      if (pendingChat === controller) pendingChat = null;
      if (currentView === viewVersion) {
        sendButton.disabled = false;
        input.focus();
      }
    }
  });
})();
