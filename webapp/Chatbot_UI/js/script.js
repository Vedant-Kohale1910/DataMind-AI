/* =======================================================================
   NEUROMORPHIC RAG CHATBOT — SCRIPT
   Sections:
     1. Theme (dark/light) toggle + persistence
     2. Icon Sphere (Data Science / ML / AI logos) — pure JS 3D projection
     3. Stage -> Chat panel "grow" transition
     4. Chat logic (send message, typing wave loader, RAG backend hook)
   ======================================================================= */

(function () {
  "use strict";

  /* =====================================================================
     1. THEME TOGGLE
     ===================================================================== */
  const root = document.documentElement;
  const themeSwitch = document.getElementById("themeSwitch");
  const STORAGE_KEY = "neurorag-theme";

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    themeSwitch.checked = theme === "light";
    localStorage.setItem(STORAGE_KEY, theme);
  }

  (function initTheme() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") {
      applyTheme(saved);
    } else {
      const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
      applyTheme(prefersLight ? "light" : "dark");
    }
  })();

  themeSwitch.addEventListener("change", () => {
    applyTheme(themeSwitch.checked ? "light" : "dark");
  });

  /* =====================================================================
     2. ICON SPHERE — Data Science / Machine Learning / AI logos only
     ===================================================================== */
  // Using the public simpleicons.org CDN (renders official brand-colored SVGs).
  const ICON_SLUGS = [
    "python", "tensorflow", "pytorch", "keras", "scikitlearn",
    "numpy", "pandas", "jupyter", "plotly", "opencv",
    "apachespark", "r", "anaconda", "nvidia", "openai",
    "googlecolab", "huggingface", "onnx"
  ];

  const sphereStage = document.getElementById("sphereStage");
  const sphereWrap = document.getElementById("sphereWrap");
  let sphereItems = [];
  let sphereRadius = 170;
  let rotX = 0.15; // slight tilt
  let rotY = 0;
  const AUTO_SPEED = 0.0032;
  let dragVX = 0, dragVY = 0;
  let isDragging = false;
  let lastPointer = { x: 0, y: 0 };

  function buildSphere() {
    sphereStage.innerHTML = "";
    sphereItems = [];
    const n = ICON_SLUGS.length;

    ICON_SLUGS.forEach((slug, i) => {
      // Even distribution on a sphere (golden section spiral)
      const phi = Math.acos(-1 + (2 * i + 1) / n);
      const theta = Math.sqrt(n * Math.PI) * phi;

      const el = document.createElement("div");
      el.className = "sphere-icon";
      const img = document.createElement("img");
      img.src = `https://cdn.simpleicons.org/${slug}`;
      img.alt = slug;
      img.loading = "lazy";
      img.onerror = () => { el.style.display = "none"; };
      el.appendChild(img);
      sphereStage.appendChild(el);

      sphereItems.push({
        el,
        theta,
        phi,
        x: 0, y: 0, z: 0
      });
    });

    sizeSphere();
  }

  function sizeSphere() {
    const size = sphereWrap.clientWidth || 460;
    sphereRadius = size * 0.37;
  }

  function renderSphere() {
    sphereItems.forEach((item) => {
      // base position on sphere
      const bx = sphereRadius * Math.sin(item.phi) * Math.cos(item.theta);
      const by = sphereRadius * Math.cos(item.phi);
      const bz = sphereRadius * Math.sin(item.phi) * Math.sin(item.theta);

      // rotate around Y axis
      let x = bx * Math.cos(rotY) + bz * Math.sin(rotY);
      let z = -bx * Math.sin(rotY) + bz * Math.cos(rotY);
      let y = by;

      // rotate around X axis
      let y2 = y * Math.cos(rotX) - z * Math.sin(rotX);
      let z2 = y * Math.sin(rotX) + z * Math.cos(rotX);

      const scale = (z2 + sphereRadius * 1.6) / (sphereRadius * 2.6);
      const opacity = 0.35 + scale * 0.75;

      item.el.style.transform = `translate3d(${x}px, ${y2}px, ${z2}px) scale(${Math.max(0.45, scale)})`;
      item.el.style.opacity = Math.min(1, opacity).toFixed(2);
      item.el.style.zIndex = Math.round(z2 + 1000);
    });
  }

  function animateSphere() {
    if (!isDragging) {
      rotY += AUTO_SPEED + dragVX;
      rotX += dragVY;
      dragVX *= 0.94;
      dragVY *= 0.94;
      // keep tilt gentle
      rotX = Math.max(-0.6, Math.min(0.6, rotX));
    }
    renderSphere();
    requestAnimationFrame(animateSphere);
  }

  // Pointer drag interaction (mouse + touch)
  function onPointerDown(e) {
    isDragging = true;
    const p = e.touches ? e.touches[0] : e;
    lastPointer = { x: p.clientX, y: p.clientY };
  }
  function onPointerMove(e) {
    if (!isDragging) return;
    const p = e.touches ? e.touches[0] : e;
    const dx = p.clientX - lastPointer.x;
    const dy = p.clientY - lastPointer.y;
    rotY += dx * 0.005;
    rotX += dy * 0.005;
    rotX = Math.max(-0.9, Math.min(0.9, rotX));
    dragVX = dx * 0.0006;
    dragVY = dy * 0.0006;
    lastPointer = { x: p.clientX, y: p.clientY };
  }
  function onPointerUp() { isDragging = false; }

  sphereWrap.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);
  sphereWrap.addEventListener("touchstart", onPointerDown, { passive: true });
  window.addEventListener("touchmove", onPointerMove, { passive: true });
  window.addEventListener("touchend", onPointerUp);

  window.addEventListener("resize", sizeSphere);

  buildSphere();
  requestAnimationFrame(animateSphere);

  /* =====================================================================
     3. STAGE -> CHAT PANEL TRANSITION
     ===================================================================== */
  const stagePage = document.getElementById("stagePage");
  const chatBackdrop = document.getElementById("chatBackdrop");
  const chatPanel = document.getElementById("chatPanel");
  const startChatBtn = document.getElementById("startChatBtn");
  const closeChatBtn = document.getElementById("closeChatBtn");
  const chatInput = document.getElementById("chatInput");

  // On mobile, the chat becomes a full-screen page, so the floating
  // top-right theme toggle would otherwise sit directly on top of the
  // header's delete/close buttons. Below ~640px, while the chat is open,
  // move the SAME toggle element into the header's action row (next to
  // the delete button) instead of leaving it floating — no duplicate
  // checkbox, so it always stays in sync with the real theme state.
  const themeToggleEl = document.getElementById("themeToggle");
  const chatActionsEl = document.querySelector(".chat-panel__actions");
  const MOBILE_MQ = window.matchMedia("(max-width: 640px)");

  function placeThemeToggle() {
    const chatOpen = chatPanel.classList.contains("is-open");
    const shouldBeInline = chatOpen && MOBILE_MQ.matches;

    if (shouldBeInline) {
      if (themeToggleEl.parentElement !== chatActionsEl) {
        chatActionsEl.insertBefore(themeToggleEl, closeChatBtn);
        themeToggleEl.classList.add("theme-toggle--inline");
      }
    } else if (themeToggleEl.parentElement !== document.body) {
      document.body.appendChild(themeToggleEl);
      themeToggleEl.classList.remove("theme-toggle--inline");
    }
  }

  let hasGreeted = false;

  function openChat() {
    stagePage.classList.add("is-hidden");
    chatBackdrop.classList.add("is-visible");
    // next frame -> trigger grow transition
    requestAnimationFrame(() => {
      chatPanel.classList.add("is-open");
      chatPanel.setAttribute("aria-hidden", "false");
      placeThemeToggle();
    });

    if (!hasGreeted) {
      hasGreeted = true;
      setTimeout(() => {
        addMessage(
          "bot",
          "Hi there 👋 Ask me anything about Machine learning and Data Science, I'll retrieve the most relevant context to solve your query"
        );
      }, 550);
    }

    setTimeout(() => chatInput.focus(), 700);
  }

  function closeChat() {
    chatPanel.classList.remove("is-open");
    chatPanel.setAttribute("aria-hidden", "true");
    chatBackdrop.classList.remove("is-visible");
    stagePage.classList.remove("is-hidden");
    placeThemeToggle();
  }

  startChatBtn.addEventListener("click", openChat);
  closeChatBtn.addEventListener("click", closeChat);
  chatBackdrop.addEventListener("click", closeChat);

  // Keep the toggle correctly placed if the viewport crosses the mobile
  // breakpoint (e.g. rotating a tablet, or resizing a desktop window)
  // while the chat happens to be open.
  MOBILE_MQ.addEventListener
    ? MOBILE_MQ.addEventListener("change", placeThemeToggle)
    : MOBILE_MQ.addListener(placeThemeToggle); // Safari <14 fallback

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && chatPanel.classList.contains("is-open")) {
      closeChat();
    }
  });

  /* =====================================================================
     4. CHAT LOGIC
     ===================================================================== */
  const chatBody = document.getElementById("chatBody");
  const chatForm = document.getElementById("chatForm");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const typingTemplate = document.getElementById("typingTemplate");

  // ---- Connect this to your real RAG backend ----
  // Point this at your API. Expected contract:
  //   POST { query: string }  ->  { answer: string }
  const RAG_API_ENDPOINT = "/api/chat";

  function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function botAvatarSVG() {
    return `<span class="logo-icon" aria-hidden="true"></span>`;
  }
  function userAvatarSVG() {
    return `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
  }

  function addMessage(role, text, sources) {
    const wrap = document.createElement("div");
    wrap.className = `msg msg--${role}`;

    const avatar = document.createElement("div");
    avatar.className = "msg__avatar" + (role === "bot" ? " bot-avatar" : "");
    avatar.innerHTML = role === "bot" ? botAvatarSVG() : userAvatarSVG();

    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";

    // Answer text — render markdown (bold, numbered/bulleted lists, etc.)
    // for bot replies, since the LLM already outputs markdown syntax.
    // User messages stay as plain text (no need to parse, and safer).
    const answerEl = document.createElement("div");
    answerEl.className = "msg__answer";
    const markdownAvailable = typeof marked !== "undefined" && typeof DOMPurify !== "undefined";
    if (role === "bot" && markdownAvailable) {
      const html = marked.parse(text, { breaks: true });
      answerEl.innerHTML = DOMPurify.sanitize(html);
    } else {
      // Falls back to plain text if the markdown CDN scripts failed to
      // load (offline, blocked by a firewall/ad-blocker, etc.) so the
      // chat still works — it just won't have rich formatting that turn.
      answerEl.textContent = text;
    }
    bubble.appendChild(answerEl);

    // Source metadata (bot messages only)
    if (role === "bot" && sources && sources.length > 0) {
      const srcBlock = document.createElement("div");
      srcBlock.className = "msg__sources";

      const srcLabel = document.createElement("span");
      srcLabel.className = "msg__sources-label";
      srcLabel.textContent = "Sources";
      srcBlock.appendChild(srcLabel);

      sources.forEach((src) => {
        const item = document.createElement("div");
        item.className = "msg__source-item";
        let line = `\u25B6 ${src.video_title}`;
        if (src.start != null && src.end != null) {
          line += `  (${src.start.toFixed(1)}s \u2013 ${src.end.toFixed(1)}s)`;
        }
        item.textContent = line;
        srcBlock.appendChild(item);
      });

      bubble.appendChild(srcBlock);
    }

    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    chatBody.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function showTyping() {
    const node = typingTemplate.content.firstElementChild.cloneNode(true);
    chatBody.appendChild(node);
    scrollToBottom();
    return node;
  }

  function removeTyping(node) {
    if (node && node.parentNode) node.parentNode.removeChild(node);
  }

  // Fallback mock responder used if no backend is connected yet (dev/demo mode).
  function mockAnswer(query) {
    const canned = [
      `Based on the retrieved context, here's what I found about "${query}": this would normally be the grounded answer from your knowledge base.`,
      `Great question. Once connected to your RAG pipeline at ${RAG_API_ENDPOINT}, I'll pull the top matching passages and summarize them here.`,
      `I've searched the vector store for relevant chunks on "${query}" — connect your retriever + LLM backend to see real, cited answers.`
    ];
    return canned[Math.floor(Math.random() * canned.length)];
  }

  async function getBotResponse(query) {
    try {
      const res = await fetch(RAG_API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      if (!res.ok) throw new Error("Bad response");
      const data = await res.json();
      return {
        answer: data.answer || mockAnswer(query),
        sources: data.sources || []
      };
    } catch (err) {
      // No backend wired up yet — fall back to a mock, simulated-latency answer.
      await new Promise((r) => setTimeout(r, 900 + Math.random() * 700));
      return { answer: mockAnswer(query), sources: [] };
    }
  }

  async function handleSend(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    chatInput.value = "";
    sendBtn.disabled = true;

    const typingNode = showTyping();
    const { answer, sources } = await getBotResponse(text);
    removeTyping(typingNode);
    addMessage("bot", answer, sources);
    sendBtn.disabled = false;
    chatInput.focus();
  }

  chatForm.addEventListener("submit", handleSend);

  clearChatBtn.addEventListener("click", () => {
    chatBody.innerHTML = "";
    hasGreeted = false;
    addMessage("bot", "Conversation cleared. How can I help you next?");
    hasGreeted = true;
  });
})();
