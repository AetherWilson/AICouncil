/* A single conversation view; council work stays available in an expandable record. */
const $ = (id) => document.getElementById(id);
const paths = {
  panel: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/>',
  edit: '<path d="M13 5H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-7M15 4l5 5M10 14l1-5 7-7 4 4-7 7z"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>',
  settings:
    '<path d="m10 3-1 3-3 1-2 3 2 2-1 3 3 3 3-1 2 2 3-1 1-3 3-2-1-3-3-1-1-3z"/><circle cx="12" cy="11" r="3"/>',
  chevron: '<path d="m7 10 5 5 5-5"/>',
  arrowUp: '<path d="M12 19V5m-6 6 6-6 6 6"/>',
  arrowDown: '<path d="M12 5v14m-6-6 6 6 6-6"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v4h16v-4"/>',
  copy: '<rect x="8" y="7" width="12" height="14" rx="2"/><path d="M15 7V3H4v13h4"/>',
  retry: '<path d="M20 10a8 8 0 1 0-1 7M20 4v6h-6"/>',
  more: '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  trash: '<path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8m-8 4h5"/>',
  compare:
    '<path d="M4 5h6M4 10h4M14 5h6m-4 5h4M7 15l-3 3 3 3m10-6 3 3-3 3M4 18h16"/>',
  spark:
    '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
function icon(name) {
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
}
function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
function element(tag, className = "", text = "") {
  const el = document.createElement(tag);
  el.className = className;
  if (text) el.textContent = text;
  return el;
}
function buttonIcon(name, title, action) {
  const button = element("button", "icon-button");
  button.innerHTML = icon(name);
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", action);
  return button;
}
function uid(prefix = "message") {
  return `${prefix}_${crypto.randomUUID ? crypto.randomUUID() : Date.now() + "_" + Math.random().toString(36).slice(2)}`;
}
const socket = io();
const conversations = {};
let currentSessionId = null;
let tempChatId = null;
let councilRoles = {};
let historyItems = [];
let currentSessionName = "新對話";
let loadSequence = 0;
let activeNodes = new Map();
let activeGroups = new Map();
let uploadsPending = 0;
let toastTimer;
const phaseNames = {
  agenda: "議程",
  proposal: "獨立提案",
  discussion: "討論",
  draft: "議長草案",
  vote: "表決",
  tally: "表決結果",
  direction: "補充指示",
  failure: "發言未完成",
};
const voteNames = {
  approve: "同意",
  request_changes: "要求修改",
  reject: "反對",
  abstain: "棄權",
  invalid: "無效票",
  unavailable: "未能投票",
};
function getActiveChatId() {
  return currentSessionId || tempChatId;
}
function ensureConversation(id) {
  if (!id) return null;
  if (!conversations[id])
    conversations[id] = {
      id,
      messages: [],
      documents: [],
      is_generating: false,
      stopping: false,
      temporary: id.startsWith("temp_"),
      name: "新對話",
      context: {},
      draft: "",
      thinking_process: { active_source: "", status: "idle" },
      pendingUpdates: [],
      renderTimers: new Map(),
    };
  return conversations[id];
}
function currentState() {
  return ensureConversation(getActiveChatId());
}
function nearBottom() {
  const view = $("conversationWindow");
  return view.scrollHeight - view.scrollTop - view.clientHeight < 140;
}
function scrollBottom(force = false) {
  if (force || nearBottom()) {
    const view = $("conversationWindow");
    view.scrollTop = view.scrollHeight;
  }
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    $("toast").hidden = true;
  }, 3000);
}
function addConsoleLog(message) {
  const log = $("consoleContent");
  log.textContent = (log.textContent + "\n" + message)
    .split("\n")
    .slice(-160)
    .join("\n");
  log.scrollTop = log.scrollHeight;
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.success === false)
    throw new Error(data.error || "無法完成操作");
  return data;
}
function post(url, body) {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
function rawText(message) {
  return message.raw_markdown ?? message.content ?? "";
}
function isUser(message) {
  return (
    message.type === "user" ||
    message.role === "user" ||
    message.council_phase === "direction" ||
    message.pending
  );
}
function isFinal(message) {
  return (
    message.is_final_response === true ||
    (message.is_final_response == null &&
      /final response/i.test(message.sender || message.bot_name || ""))
  );
}
function isIntermediate(message) {
  return (
    !isUser(message) &&
    !isFinal(message) &&
    !!(
      message.run_group_id ||
      message.event_kind?.startsWith("council_") ||
      message.is_council_intermediate
    )
  );
}
function answerText(message) {
  const raw = rawText(message);
  if (message.council_phase !== "final" || !raw.startsWith("> **議會表決："))
    return raw;
  let answer = raw.slice(raw.indexOf("\n\n") + 2);
  const marker = "\n\n---\n\n**表決理由 / Ballot reasons**";
  const split = answer.lastIndexOf(marker);
  if (split >= 0) answer = answer.slice(0, split);
  return answer;
}
function renderMarkdown(target, text, streaming = false) {
  if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
    target.textContent = text;
    return;
  }
  target.innerHTML = DOMPurify.sanitize(
    marked.parse(text, { breaks: false, gfm: true }),
  );
  target.querySelectorAll("a").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  target.querySelectorAll("pre").forEach((pre) => {
    const shell = element("div", "code-shell");
    const header = element("div", "code-header");
    const code = pre.querySelector("code");
    const language = code?.className.match(/language-([\w-]+)/)?.[1] || "text";
    header.append(element("span", "", language));
    const copy = element("button", "", "複製程式碼");
    copy.addEventListener("click", () =>
      copyText(code?.textContent || pre.textContent),
    );
    header.append(copy);
    pre.replaceWith(shell);
    shell.append(header, pre);
    if (!streaming && code && typeof hljs !== "undefined") {
      try {
        hljs.highlightElement(code);
      } catch (_) {
        /* Unknown languages remain plain text. */
      }
    }
  });
  target.querySelectorAll("table").forEach((table) => {
    const wrap = element("div", "table-scroll");
    table.replaceWith(wrap);
    wrap.append(table);
  });
  if (!streaming && typeof renderMathInElement !== "undefined")
    renderMathInElement(target, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
}
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已複製");
  } catch (_) {
    toast("瀏覽器無法存取剪貼簿");
  }
}
function attachmentChip(doc, removable = false) {
  const chip = element("div", "attachment-chip");
  const mark = element("span");
  mark.innerHTML = icon("file");
  const name = element("div", "attachment-name", doc.filename);
  name.title = doc.filename;
  name.append(
    element(
      "small",
      "attachment-type",
      doc.type === "image"
        ? "圖片"
        : doc.type === "word"
          ? "Word 文件"
          : "PDF 文件",
    ),
  );
  chip.append(mark, name);
  if (removable)
    chip.append(
      buttonIcon("close", `移除 ${doc.filename}`, async () => {
        const state = currentState();
        if (!state) return;
        state.documents = state.documents.filter(
          (item) => item.filename !== doc.filename,
        );
        renderAttachments();
        await post("/api/remove_document", {
          chat_id: state.id,
          filename: doc.filename,
        }).catch((e) => toast(e.message));
        saveState(state);
      }),
    );
  return chip;
}
function renderAttachments() {
  const list = $("attachmentList");
  list.replaceChildren();
  (currentState()?.documents || []).forEach((doc) =>
    list.append(attachmentChip(doc, true)),
  );
}
function groupNode(id, state) {
  if (activeGroups.has(id)) return activeGroups.get(id);
  const details = element(
    "details",
    "council-discussion thinking-process-container",
  );
  details.dataset.runGroupId = id;
  const summary = element("summary");
  summary.append(element("span", "discussion-label", "議會討論"));
  const status = element("span", "discussion-status");
  const chevron = element("span", "discussion-chevron");
  chevron.innerHTML = icon("chevron");
  summary.append(status, chevron);
  const body = element("div", "discussion-body");
  details.append(summary, body);
  $("messageList").append(details);
  activeGroups.set(id, details);
  updateGroup(details, state);
  return details;
}
function updateGroup(group, state) {
  const running =
    state.is_generating && group.dataset.runGroupId === state.run_group_id;
  const status = group.querySelector(".discussion-status");
  status.replaceChildren();
  if (running && !state.stopping) {
    status.append(element("span", "status-spinner"));
    status.append(
      document.createTextNode(
        " " + (state.phase ? phaseNames[state.phase] || "進行中" : "進行中"),
      ),
    );
  } else
    status.textContent =
      state.stopping || state.thinking_process.status === "stopped"
        ? "已停止"
        : "已完成";
}
function renderMessage(message, state) {
  const wasBottom = nearBottom();
  let node = activeNodes.get(message.id);
  const user = isUser(message);
  if (!node) {
    node = element("article", "message " + (user ? "user" : "ai"));
    node.dataset.messageId = message.id;
    activeNodes.set(message.id, node);
    if (isIntermediate(message))
      groupNode(message.run_group_id || "legacy", state)
        .querySelector(".discussion-body")
        .append(node);
    else $("messageList").append(node);
  }
  node.dataset.isFinalResponse = isFinal(message) ? "true" : "false";
  node.dataset.eventKind = message.event_kind || "";
  node.dataset.councilPhase = message.council_phase || "";
  node.replaceChildren();
  if (!user) {
    const header = element("div", "message-header");
    if (isFinal(message)) {
      header.append(
        element("span", "assistant-avatar", "◎"),
        element("span", "assistant-name", "AI Council"),
      );
    } else {
      const member = councilRoles[message.role_name];
      header.append(
        element(
          "span",
          "",
          member?.name || message.sender || message.bot_name || "AI",
        ),
      );
      header.append(
        element(
          "span",
          "phase-label",
          `${phaseNames[message.council_phase] || ""}${message.council_round ? " · " + message.council_round : ""}`,
        ),
      );
    }
    header.title = message.sender || message.bot_name || "";
    if (
      isFinal(message) &&
      typeof message.council_payload?.approved === "boolean"
    ) {
      const votes = message.council_payload.ballots || [];
      const approved = message.council_payload.approved;
      const count = votes.filter((ballot) => ballot.vote === "approve").length;
      header.append(
        element(
          "span",
          "ballot-badge" + (approved ? "" : " unapproved"),
          `${approved ? "議會通過" : "尚未通過"}${votes.length ? " · " + count + "/" + votes.length : ""}`,
        ),
      );
    }
    node.append(header);
  }
  if (user && message.attachments?.length) {
    const sources = element("div", "message-attachments");
    message.attachments.forEach((doc) => sources.append(attachmentChip(doc)));
    node.append(sources);
  }
  const content = element("div", "message-content");
  if (user) content.textContent = rawText(message);
  else if (message.stream_status === "running" && !rawText(message))
    content.classList.add("streaming-content");
  else
    renderMarkdown(
      content,
      isFinal(message) ? answerText(message) : rawText(message),
      message.stream_status === "running",
    );
  node.append(content);
  if (message.pending)
    node.append(element("small", "phase-label", "將於下一個階段套用"));
  if (isFinal(message) && message.council_payload?.ballots?.length) {
    const details = element("details", "ballot-details");
    const summary = element("summary");
    summary.innerHTML = icon("chevron");
    summary.append(document.createTextNode("查看表決理由"));
    details.append(summary);
    message.council_payload.ballots.forEach((ballot) => {
      const row = element("div", "ballot-row");
      row.append(
        element(
          "strong",
          "",
          `${ballot.name} · ${voteNames[ballot.vote] || ballot.vote}`,
        ),
        element("span", "", ballot.reason),
      );
      details.append(row);
    });
    node.append(details);
  }
  if (
    message.stream_status !== "running" &&
    !message.pending &&
    (!isIntermediate(message) || user)
  ) {
    const actions = element("div", "message-actions");
    actions.append(
      buttonIcon("copy", "複製訊息", () =>
        copyText(user ? rawText(message) : answerText(message)),
      ),
    );
    if (user)
      actions.append(
        buttonIcon("edit", "編輯訊息並重新討論", () =>
          editMessage(message, state),
        ),
      );
    else if (isFinal(message))
      actions.append(
        buttonIcon("retry", "重新討論", () => regenerate(message, state)),
      );
    node.append(actions);
  }
  if (wasBottom) scrollBottom(true);
}
function renderConversation() {
  activeNodes = new Map();
  activeGroups = new Map();
  $("messageList").replaceChildren();
  const state = currentState();
  $("welcome").hidden = !!state?.messages.length;
  (state?.messages || []).forEach((message) => renderMessage(message, state));
  $("currentSessionName").textContent = state?.name || "新對話";
  document.title = state?.messages.length
    ? `${state.name} · AI Council`
    : "AI Council";
  $("messageInput").value = state?.draft || "";
  resizeComposer();
  renderAttachments();
  setProcessingState();
  renderContext();
  scrollBottom(true);
}
function setProcessingState() {
  const state = currentState();
  const running = !!state?.is_generating;
  const send = $("sendButton");
  send.innerHTML = icon(running ? "stop" : "arrowUp");
  send.setAttribute(
    "aria-label",
    state?.stopping ? "正在停止" : running ? "停止回覆" : "傳送訊息",
  );
  send.title = send.getAttribute("aria-label");
  send.disabled =
    !!state?.stopping ||
    (!running &&
      (!socket.connected ||
        uploadsPending > 0 ||
        !$("messageInput").value.trim()));
  send.classList.toggle("stopping", !!state?.stopping);
  $("councilUpdateButton").hidden = !running || !!state?.stopping;
  $("attachButton").disabled =
    running || uploadsPending > 0 || !socket.connected;
  $("councilUpdateButton").disabled =
    !socket.connected || !$("messageInput").value.trim();
  $("messageInput").placeholder = running
    ? "補充指示，按 Enter 送出…"
    : "傳送訊息給 AI Council";
  if (!running)
    $("liveStatus").textContent = "AI 也可能出錯，重要資訊請再次確認。";
  else if (
    !["compacting", "continuing", "recovering", "queued"].includes(
      state.context.status,
    )
  )
    $("liveStatus").textContent = state.stopping
      ? "正在停止回覆…"
      : "議會正在依序討論…";
  activeGroups.forEach((group) => updateGroup(group, state));
}
function ensureChat() {
  if (!getActiveChatId()) {
    currentSessionId = uid("chat");
    currentSessionName = "新對話";
  }
  const state = currentState();
  socket.emit("join_chat", { chat_id: state.id });
  return state;
}
function getCurrentMessages() {
  return (currentState()?.messages || [])
    .filter((message) => !message.pending)
    .map((message) => ({ ...message }));
}
function saveState(state) {
  if (!state || state.deleted || !state.messages.some((m) => !m.pending))
    return;
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(async () => {
    try {
      await post(state.temporary ? "/api/save_temp_chat" : "/api/save_chat", {
        id: state.id,
        name: state.name,
        messages: state.messages.filter((m) => !m.pending),
        uploadedDocuments: state.documents,
      });
      refreshHistory();
    } catch (error) {
      addConsoleLog(error.message);
      toast("對話尚未儲存，請檢查本機服務。");
    }
  }, 350);
}
function startNewChat() {
  loadSequence++;
  currentSessionId = null;
  tempChatId = null;
  currentSessionName = "新對話";
  localStorage.removeItem("council.activeChat");
  renderConversation();
  renderHistory();
  closeMobileSidebar();
  $("messageInput").focus();
}
function sendMessage() {
  const text = $("messageInput").value.trim();
  const prior = currentState();
  if (prior?.is_generating) {
    if (text) sendCouncilUpdate();
    return;
  }
  if (!text || !socket.connected || uploadsPending) return;
  const state = ensureChat();
  if (!state.messages.length)
    state.name = text.replace(/\s+/g, " ").slice(0, 48);
  const message = {
    id: uid(),
    type: "user",
    role: "user",
    content: text,
    raw_markdown: text,
    sender: "You",
    attachments: state.documents.map((doc) => ({ ...doc })),
  };
  state.messages.push(message);
  state.is_generating = true;
  state.stopping = false;
  state.draft = "";
  state.thinking_process.status = "running";
  $("welcome").hidden = true;
  $("currentSessionName").textContent = state.name;
  $("messageInput").value = "";
  resizeComposer();
  renderMessage(message, state);
  setProcessingState();
  scrollBottom(true);
  localStorage.setItem("council.activeChat", state.id);
  socket.emit("load_chat_history", {
    chat_id: state.id,
    messages: state.messages,
  });
  socket.emit("send_message", {
    chat_id: state.id,
    message: text,
    existing_user_message_id: message.id,
    attachments: message.attachments,
    reset_documents: true,
  });
  saveState(state);
  refreshHistory();
}
function sendCouncilUpdate() {
  const text = $("messageInput").value.trim();
  const state = currentState();
  if (!state?.is_generating || state.stopping || !text || !socket.connected)
    return;
  const message = {
    id: uid("pending"),
    type: "user",
    content: text,
    raw_markdown: text,
    pending: true,
  };
  state.messages.push(message);
  state.pendingUpdates.push(message.id);
  renderMessage(message, state);
  socket.emit("send_message", { chat_id: state.id, message: text });
  $("messageInput").value = "";
  state.draft = "";
  resizeComposer();
  scrollBottom(true);
}
function stopAI() {
  const state = currentState();
  if (!state?.is_generating || state.stopping) return;
  state.stopping = true;
  state.thinking_process.status = "stopped";
  socket.emit("stop_generation", { chat_id: state.id });
  setProcessingState();
}
function regenerate(message, state) {
  if (state.is_generating) {
    toast("請先停止目前的討論");
    return;
  }
  const at = state.messages.findIndex((item) => item.id === message.id);
  let userIndex = at - 1;
  while (userIndex >= 0 && state.messages[userIndex].type !== "user")
    userIndex--;
  if (userIndex < 0) return;
  rerunFromUser(state, userIndex);
}
function rerunFromUser(state, index) {
  const user = state.messages[index];
  state.messages = state.messages.slice(0, index + 1);
  state.documents = (user.attachments || []).map((doc) => ({ ...doc }));
  state.is_generating = true;
  state.stopping = false;
  state.thinking_process.status = "running";
  state.context.status = "ready";
  renderConversation();
  socket.emit("join_chat", { chat_id: state.id });
  socket.emit("load_chat_history", {
    chat_id: state.id,
    messages: state.messages,
  });
  socket.emit("send_message", {
    chat_id: state.id,
    message: rawText(user),
    existing_user_message_id: user.id,
    attachments: user.attachments || [],
    reset_documents: true,
  });
  saveState(state);
}
function editMessage(message, state) {
  if (state.is_generating) {
    toast("請先停止目前的討論");
    return;
  }
  const node = activeNodes.get(message.id);
  const form = element("form", "edit-message");
  const textarea = element("textarea");
  textarea.value = rawText(message);
  textarea.setAttribute("aria-label", "編輯訊息");
  const actions = element("div", "edit-actions");
  const cancel = element("button", "secondary-button", "取消");
  cancel.type = "button";
  cancel.onclick = () => renderMessage(message, state);
  const submit = element("button", "primary-button", "儲存並重新討論");
  submit.type = "submit";
  actions.append(cancel, submit);
  form.append(textarea, actions);
  node.replaceChildren(form);
  textarea.focus();
  form.onsubmit = (event) => {
    event.preventDefault();
    const value = textarea.value.trim();
    if (!value) return;
    const index = state.messages.findIndex((item) => item.id === message.id);
    if (
      index < state.messages.length - 1 &&
      !confirm("將移除這則訊息之後的回覆，並重新討論。繼續嗎？")
    )
      return;
    message.content = value;
    message.raw_markdown = value;
    rerunFromUser(state, index);
  };
}
async function uploadFiles(files) {
  if (currentState()?.is_generating) {
    toast("請等待討論結束後再附加檔案");
    return;
  }
  const state = ensureChat();
  for (const file of files) {
    if (!/\.(pdf|jpe?g|png|docx|docm|dotx|dotm)$/i.test(file.name)) {
      toast("請選擇 PDF、圖片或 Word OpenXML 文件");
      continue;
    }
    if (file.size > 16 * 1024 * 1024) {
      toast("每個檔案上限為 16 MB");
      continue;
    }
    uploadsPending++;
    $("uploadStatus").textContent = `正在上傳 ${file.name}…`;
    setProcessingState();
    try {
      const body = new FormData();
      body.append("chat_id", state.id);
      body.append("file", file);
      const data = await api("/api/upload_document", { method: "POST", body });
      state.documents = state.documents.filter(
        (doc) => doc.filename !== data.document.filename,
      );
      state.documents.push(data.document);
    } catch (error) {
      toast(error.message);
    } finally {
      uploadsPending--;
      if (state.id === getActiveChatId()) renderAttachments();
      $("uploadStatus").textContent = uploadsPending ? "正在上傳…" : "";
      setProcessingState();
    }
  }
  saveState(state);
  $("pdfFileInput").value = "";
}
function resizeComposer() {
  const input = $("messageInput");
  input.style.height = "auto";
  input.style.height = Math.min(200, input.scrollHeight) + "px";
}
function toggleSidebar() {
  if (matchMedia("(max-width: 800px)").matches)
    $("appShell").classList.toggle("mobile-sidebar-open");
  else {
    $("appShell").classList.toggle("sidebar-collapsed");
    localStorage.setItem(
      "council.sidebarCollapsed",
      $("appShell").classList.contains("sidebar-collapsed") ? "true" : "false",
    );
  }
}
function closeMobileSidebar() {
  $("appShell").classList.remove("mobile-sidebar-open");
}
async function refreshHistory() {
  try {
    const [saved, temporary] = await Promise.all([
      api("/api/list_chats"),
      api("/api/list_temp_chats"),
    ]);
    historyItems = [
      ...saved.chats.map((chat) => ({ ...chat, temporary: false })),
      ...temporary.chats
        .filter((chat) => !saved.chats.some((item) => item.id === chat.id))
        .map((chat) => ({ ...chat, temporary: true })),
    ].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    renderHistory();
  } catch (error) {
    addConsoleLog(error.message);
  }
}
function renderHistory() {
  const target = $("historyList");
  const query = $("historySearch").value.toLowerCase();
  target.replaceChildren();
  const visible = historyItems.filter((chat) =>
    String(chat.name).toLowerCase().includes(query),
  );
  if (!visible.length) {
    target.append(
      element(
        "div",
        "history-empty",
        query ? "沒有符合的對話" : "從一個問題開始。\n對話會自動保留在這裡。",
      ),
    );
    return;
  }
  let previousHeading = "";
  visible.forEach((chat) => {
    const date = new Date(chat.timestamp);
    const days = Math.floor((Date.now() - date.getTime()) / 86400000);
    const heading = days < 1 ? "最近" : days < 7 ? "過去 7 天" : "較早的對話";
    if (heading !== previousHeading) {
      target.append(element("div", "history-heading", heading));
      previousHeading = heading;
    }
    const row = element(
      "div",
      "history-item" + (chat.id === getActiveChatId() ? " active" : ""),
    );
    const open = element("button", "history-open", chat.name || "未命名對話");
    open.title = chat.name || "未命名對話";
    open.onclick = () => loadChat(chat.id, chat.temporary);
    const menu = buttonIcon(
      "more",
      `管理對話：${chat.name || "未命名對話"}`,
      () => manageChat(chat),
    );
    menu.classList.add("history-menu");
    row.append(open, menu);
    target.append(row);
  });
}
async function loadChat(
  id,
  temporary = id.startsWith("temp_"),
  reconcile = false,
) {
  const sequence = ++loadSequence;
  try {
    const data = await api(
      `${temporary ? "/api/load_temp_chat/" : "/api/load_chat/"}${encodeURIComponent(id)}`,
    );
    if (sequence !== loadSequence) return;
    const state = ensureConversation(id);
    const live = state.is_generating;
    if (!live || reconcile)
      state.messages = (data.chat.messages || [])
        .filter(
          (msg) =>
            !(
              msg.type === "ai" &&
              msg.sender === "System" &&
              /Welcome to|New chat started|history cleared/i.test(rawText(msg))
            ),
        )
        .map((message) => ({
          ...message,
          id: message.id || uid(),
          stream_status:
            message.stream_status === "running" && !data.chat.isGenerating
              ? "stopped"
              : message.stream_status,
        }));
    state.is_generating = reconcile
      ? !!data.chat.isGenerating
      : live || !!data.chat.isGenerating;
    state.name = data.chat.name || "未命名對話";
    state.temporary = temporary;
    if (!live || reconcile) {
      state.context = data.chat.context_usage || {};
      state.documents = data.chat.uploadedDocuments || [];
    }
    currentSessionId = temporary ? null : id;
    tempChatId = temporary ? id : null;
    currentSessionName = state.name;
    localStorage.setItem("council.activeChat", id);
    socket.emit("join_chat", { chat_id: id });
    renderConversation();
    renderHistory();
    closeMobileSidebar();
    if (state.documents.length)
      await post("/api/restore_documents", {
        chat_id: id,
        documents: state.documents,
      });
  } catch (error) {
    if (sequence === loadSequence) toast(error.message);
  }
}
function loadTempChat(id) {
  return loadChat(id, true);
}
function manageChat(chat) {
  const dialog = element("dialog", "context-dialog");
  const header = element("div", "dialog-header");
  header.append(
    element("h2", "", "對話選項"),
    buttonIcon("close", "關閉", () => dialog.close()),
  );
  const form = element("form", "rename-form");
  const label = element("label", "", "對話名稱");
  const input = element("input", "rename-input");
  input.value = chat.name || "";
  input.required = true;
  input.maxLength = 120;
  input.setAttribute("aria-label", "對話名稱");
  label.append(input);
  const save = element("button", "primary-button", "儲存名稱");
  save.type = "submit";
  const remove = element(
    "button",
    "secondary-button delete-chat-button",
    "刪除對話",
  );
  remove.type = "button";
  form.append(label, save);
  dialog.append(header, form, remove);
  document.body.append(dialog);
  dialog.showModal();
  input.focus();
  input.select();
  dialog.addEventListener("close", () => dialog.remove());
  form.onsubmit = async (event) => {
    event.preventDefault();
    const name = input.value.trim();
    if (!name) return;
    if (conversations[chat.id]?.is_generating) {
      toast("請等待這段對話的回覆完成");
      return;
    }
    try {
      clearTimeout(conversations[chat.id]?.saveTimer);
      const data = await api(
        `${chat.temporary ? "/api/load_temp_chat/" : "/api/load_chat/"}${encodeURIComponent(chat.id)}`,
      );
      await post("/api/save_chat", { ...data.chat, name });
      if (chat.temporary) await post("/api/delete_temp_chat", { id: chat.id });
      if (conversations[chat.id]) {
        conversations[chat.id].name = name;
        conversations[chat.id].temporary = false;
      }
      if (getActiveChatId() === chat.id) {
        currentSessionId = chat.id;
        tempChatId = null;
        $("currentSessionName").textContent = name;
      }
      dialog.close();
      await refreshHistory();
    } catch (error) {
      if (conversations[chat.id]) conversations[chat.id].deleted = false;
      toast(error.message);
    }
  };
  remove.onclick = async () => {
    if (conversations[chat.id]?.is_generating) {
      toast("請先停止這段對話的回覆");
      return;
    }
    if (!confirm("確定刪除這段對話？")) return;
    try {
      clearTimeout(conversations[chat.id]?.saveTimer);
      if (conversations[chat.id]) conversations[chat.id].deleted = true;
      await post(
        chat.temporary ? "/api/delete_temp_chat" : "/api/delete_chat",
        { id: chat.id },
      );
      delete conversations[chat.id];
      if (getActiveChatId() === chat.id) startNewChat();
      dialog.close();
      await refreshHistory();
    } catch (error) {
      if (conversations[chat.id]) conversations[chat.id].deleted = false;
      toast(error.message);
    }
  };
}
function renderContext() {
  const context = currentState()?.context || {};
  const percent = context.context_window_tokens
    ? Math.min(
        100,
        Math.ceil(
          ((context.input_tokens || 0) / context.context_window_tokens) * 100,
        ),
      )
    : null;
  $("contextLabel").textContent =
    percent === null ? "上下文" : `上下文 ${percent}%`;
  $("contextPercent").textContent = percent === null ? "—" : `${percent}%`;
  $("contextProgress").value = percent || 0;
  $("contextDescription").textContent =
    percent === null
      ? "開始對話後，會顯示最近一次請求的預估用量。"
      : `約 ${(context.input_tokens || 0).toLocaleString()} / ${context.context_window_tokens.toLocaleString()} tokens。原始附件的實際用量可能因模型而異。`;
  $("checkpointCount").textContent = context.compactions
    ? `已建立 ${context.compactions} 次脈絡檢查點，儲存後可繼續使用。`
    : "尚未需要整理較早的脈絡。";
  $("compactButton").disabled = !getActiveChatId();
  $("contextRing").style.borderColor =
    percent > 75 ? "var(--danger)" : "var(--muted)";
}
function openSettings(tab = "council") {
  document
    .querySelectorAll("[data-settings-tab]")
    .forEach((button) =>
      button.setAttribute(
        "aria-selected",
        String(button.dataset.settingsTab === tab),
      ),
    );
  ["council", "memory", "backend"].forEach((name) => {
    $("settings" + name[0].toUpperCase() + name.slice(1)).hidden = name !== tab;
  });
  if (!$("settingsDialog").open) $("settingsDialog").showModal();
  if (tab === "memory") loadMemories();
}
async function loadCouncil() {
  try {
    councilRoles = await api("/api/council");
    const list = $("councilRolesList");
    list.replaceChildren();
    Object.entries(councilRoles)
      .sort((a, b) => a[1].order - b[1].order)
      .forEach(([id, seat]) => {
        const row = element("div", "council-role-item");
        row.dataset.role = id;
        const info = element("div", "council-role-info");
        info.append(
          element(
            "div",
            "council-role-name",
            seat.name + (seat.is_chair ? " · 議長" : ""),
          ),
          element("div", "council-role-model", seat.model_name),
        );
        const light = element("span", "council-role-light");
        light.dataset.role = id;
        row.append(
          element("span", "seat-number", String(seat.order).padStart(2, "0")),
          info,
          light,
        );
        list.append(row);
      });
  } catch (error) {
    toast(error.message);
    addConsoleLog(error.message);
  }
}
async function loadMemories() {
  try {
    const data = await api("/api/memories");
    const list = $("memoryList");
    list.replaceChildren();
    data.memories.forEach((memory) => {
      const row = element("div", "memory-row");
      const body = element("div");
      body.append(
        element("small", "", memory.section),
        element("p", "", memory.content),
      );
      row.append(
        body,
        buttonIcon("trash", "刪除記憶", async () => {
          await api(`/api/memories/${memory.id}`, { method: "DELETE" }).catch(
            (error) => toast(error.message),
          );
          loadMemories();
        }),
      );
      list.append(row);
    });
    if (!data.memories.length)
      list.append(element("p", "muted", "尚未新增記憶。"));
  } catch (error) {
    toast(error.message);
  }
}
async function runUptest() {
  $("uptestRunButton").disabled = true;
  const target = $("uptestResults");
  target.textContent = "正在測試模型連線…";
  try {
    const response = await fetch("/api/backend/uptest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) throw new Error("連線測試失敗");
    const text = await response.text();
    target.replaceChildren();
    text
      .trim()
      .split("\n")
      .forEach((line) => {
        try {
          const event = JSON.parse(line);
          const row = event.result || event;
          if (row.model_id)
            target.append(
              element(
                "div",
                "",
                `${row.role_label || row.role || row.model_id} · ${row.status}${row.elapsed_ms != null ? " · " + row.elapsed_ms + " ms" : ""}${row.error ? " · " + row.error : ""}`,
              ),
            );
        } catch (_) {
          /* Ignore keepalive lines. */
        }
      });
  } catch (error) {
    target.textContent = error.message;
  } finally {
    $("uptestRunButton").disabled = false;
  }
}
function upsertPacket(data, ended = false) {
  const state = ensureConversation(data.chat_id);
  if (!state) return;
  let message = state.messages.find((item) => item.id === data.message_id);
  if (!message) {
    if (data.council_phase === "direction") {
      const pending = state.messages.find(
        (item) => item.pending && rawText(item) === (data.message || ""),
      );
      if (pending && ended) {
        state.messages = state.messages.filter((item) => item !== pending);
        activeNodes.get(pending.id)?.remove();
        activeNodes.delete(pending.id);
      }
    }
    message = {
      id: data.message_id,
      type: "ai",
      role: "assistant",
      content: "",
      raw_markdown: "",
    };
    state.messages.push(message);
  }
  Object.assign(message, {
    sender: data.bot_name,
    bot_name: data.bot_name,
    bot_id: data.bot_id,
    role_name: data.role_name,
    model_id: data.model_id,
    run_group_id: data.run_group_id,
    run_id: data.run_id,
    event_kind: data.event_kind,
    council_phase: data.council_phase,
    council_round: data.council_round,
    council_payload: data.council_payload,
    is_final_response: !!data.is_final_response,
    stream_status: ended
      ? data.stream_status || (data.stopped ? "stopped" : "done")
      : "running",
  });
  if (ended) {
    message.content = data.message || "";
    message.raw_markdown = data.message || "";
  }
  if (state.id === getActiveChatId()) {
    $("welcome").hidden = true;
    renderMessage(message, state);
  }
  if (ended) saveState(state);
}
socket.on("connect", () => {
  $("connectionStatus").classList.add("connected");
  $("connectionStatus").title = "已連線";
  $("connectionStatus").setAttribute("aria-label", "已連線");
  Object.keys(conversations).forEach((id) =>
    socket.emit("join_chat", { chat_id: id }),
  );
  const active = currentState();
  if (active) loadChat(active.id, active.temporary, true);
  setProcessingState();
});
socket.on("disconnect", () => {
  $("connectionStatus").classList.remove("connected");
  $("connectionStatus").title = "連線中斷，正在重新連線";
  $("connectionStatus").setAttribute("aria-label", "連線中斷");
  setProcessingState();
});
socket.on("run_group_start", (data) => {
  const state = ensureConversation(data.chat_id);
  state.run_group_id = data.run_group_id;
  state.is_generating = true;
  state.stopping = false;
  state.thinking_process.status = "running";
  if (state.id === getActiveChatId()) setProcessingState();
});
socket.on("ai_response_start", (data) => upsertPacket(data));
socket.on("ai_response_chunk", (data) => {
  const state = ensureConversation(data.chat_id);
  const message = state?.messages.find((item) => item.id === data.message_id);
  if (!message) {
    upsertPacket(data);
    const recovered = state.messages.find(
      (item) => item.id === data.message_id,
    );
    if (recovered) {
      recovered.content = data.chunk || "";
      recovered.raw_markdown = recovered.content;
    }
    return;
  }
  message.content += data.chunk || "";
  message.raw_markdown = message.content;
  if (state.id !== getActiveChatId() || state.renderTimers.has(message.id))
    return;
  state.renderTimers.set(
    message.id,
    setTimeout(() => {
      state.renderTimers.delete(message.id);
      if (state.id === getActiveChatId()) renderMessage(message, state);
    }, 120),
  );
});
socket.on("ai_response_end", (data) => {
  const state = ensureConversation(data.chat_id);
  const timer = state.renderTimers.get(data.message_id);
  if (timer) {
    clearTimeout(timer);
    state.renderTimers.delete(data.message_id);
  }
  upsertPacket(data, true);
  if (data.council_phase === "direction") {
    state.messages
      .filter((message) => message.pending && rawText(message) === data.message)
      .forEach((message) => {
        state.messages = state.messages.filter((item) => item !== message);
        activeNodes.get(message.id)?.remove();
        activeNodes.delete(message.id);
      });
  }
});
socket.on("council_status", (data) => {
  const state = ensureConversation(data.chat_id);
  state.phase = data.council_phase;
  state.thinking_process.active_source = `${councilRoles[data.role]?.name || data.role} · ${data.council_phase || ""}`;
  if (state.id === getActiveChatId()) {
    document.querySelectorAll(".council-role-light").forEach((light) => {
      if (light.dataset.role === data.role)
        light.className = "council-role-light " + data.status;
    });
    activeGroups.forEach((group) => updateGroup(group, state));
  }
});
socket.on("context_status", (data) => {
  const state = ensureConversation(data.chat_id);
  state.context = { ...state.context, ...data };
  if (state.id !== getActiveChatId()) return;
  const messages = {
    compacting: "正在整理較早的對話脈絡…",
    compacted: "已整理脈絡，繼續處理…",
    continuing: "正在接續長回覆…",
    recovering: "正在調整上下文並接續…",
    queued: "將於下一次回覆前整理脈絡",
  };
  if (messages[data.status])
    $("liveStatus").textContent = messages[data.status];
  if (data.status === "compacted") toast("已整理上下文，完整對話仍然保留");
  renderContext();
});
socket.on("console_log", (data) => {
  addConsoleLog(data.message || "");
});
socket.on("run_group_end", (data) => {
  if (data.stopped)
    ensureConversation(data.chat_id).thinking_process.status = "stopped";
});
socket.on("all_done", (data) => {
  const state = ensureConversation(data.chat_id);
  state.is_generating = false;
  state.stopping = false;
  state.thinking_process.status =
    state.thinking_process.status === "stopped" ? "stopped" : "done";
  state.context.status = "ready";
  if (state.id === getActiveChatId()) {
    activeGroups.forEach((group) => updateGroup(group, state));
    setProcessingState();
  }
  saveState(state);
  refreshHistory();
});

hydrateIcons();
if (localStorage.getItem("council.sidebarCollapsed") === "true")
  $("appShell").classList.add("sidebar-collapsed");
$("openSidebar").onclick = toggleSidebar;
$("closeSidebar").onclick = toggleSidebar;
$("sidebarScrim").onclick = closeMobileSidebar;
$("newChatButton").onclick = startNewChat;
$("historySearch").oninput = renderHistory;
$("settingsButton").onclick = () => openSettings();
$("councilButton").onclick = () => openSettings();
$("contextButton").onclick = () => {
  renderContext();
  $("contextDialog").showModal();
};
$("compactButton").onclick = () => {
  const state = currentState();
  if (!state) return;
  socket.emit("join_chat", { chat_id: state.id });
  socket.emit("compact_context", { chat_id: state.id });
  $("contextDialog").close();
  toast("下一次模型回覆前會整理較早的脈絡");
};
$("themeButton").onclick = () => {
  const theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("council.theme", theme);
};
$("exportButton").onclick = () => {
  const state = currentState();
  if (!state?.messages.length) return;
  const text = state.messages
    .filter((m) => !m.pending)
    .map(
      (m) =>
        `## ${isUser(m) ? "You" : m.sender || "AI Council"}\n\n${rawText(m)}`,
    )
    .join("\n\n---\n\n");
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/markdown;charset=utf-8" }),
  );
  const link = element("a");
  link.href = url;
  link.download = "AI-Council.md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("sendButton").onclick = () => {
  if (currentState()?.is_generating) stopAI();
  else sendMessage();
};
$("councilUpdateButton").onclick = sendCouncilUpdate;
$("attachButton").onclick = () => $("pdfFileInput").click();
$("pdfFileInput").onchange = (event) => uploadFiles([...event.target.files]);
$("messageInput").addEventListener("input", () => {
  if (currentState()) currentState().draft = $("messageInput").value;
  resizeComposer();
  setProcessingState();
});
$("messageInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage();
  }
});
$("messageInput").addEventListener("paste", (event) => {
  if (event.clipboardData.files.length) {
    event.preventDefault();
    uploadFiles([...event.clipboardData.files]);
  }
});
$("messageDropZone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("messageDropZone").classList.add("dragover");
});
$("messageDropZone").addEventListener("dragleave", () =>
  $("messageDropZone").classList.remove("dragover"),
);
$("messageDropZone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("messageDropZone").classList.remove("dragover");
  uploadFiles([...event.dataTransfer.files]);
});
$("conversationWindow").addEventListener("scroll", () => {
  $("jumpBottom").hidden = nearBottom();
});
$("jumpBottom").onclick = () => scrollBottom(true);
document.querySelectorAll("[data-suggestion]").forEach((button) => {
  button.onclick = () => {
    $("messageInput").value = button.dataset.suggestion;
    resizeComposer();
    setProcessingState();
    $("messageInput").focus();
  };
});
document.querySelectorAll("[data-settings-tab]").forEach((button) => {
  button.onclick = () => openSettings(button.dataset.settingsTab);
});
document.querySelectorAll(".close-dialog").forEach((button) => {
  button.onclick = () => button.closest("dialog").close();
});
document.querySelectorAll("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    const box = dialog.getBoundingClientRect();
    if (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    )
      dialog.close();
  }),
);
$("memoryForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    await post("/api/memories", {
      section: $("memorySection").value,
      content: $("memoryContent").value,
    });
    $("memoryContent").value = "";
    loadMemories();
  } catch (error) {
    toast(error.message);
  }
};
$("uptestRunButton").onclick = runUptest;
document.addEventListener("keydown", (event) => {
  if (
    (event.ctrlKey || event.metaKey) &&
    event.shiftKey &&
    event.key.toLowerCase() === "o"
  ) {
    event.preventDefault();
    startNewChat();
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (matchMedia("(max-width: 800px)").matches)
      $("appShell").classList.add("mobile-sidebar-open");
    else $("appShell").classList.remove("sidebar-collapsed");
    $("historySearch").focus();
  }
});
window.addEventListener("beforeunload", () => {
  Object.values(conversations)
    .filter((state) => state.messages.length && !state.is_generating)
    .forEach((state) =>
      navigator.sendBeacon(
        state.temporary ? "/api/save_temp_chat" : "/api/save_chat",
        new Blob(
          [
            JSON.stringify({
              id: state.id,
              name: state.name,
              messages: state.messages.filter((m) => !m.pending),
              uploadedDocuments: state.documents,
            }),
          ],
          { type: "application/json" },
        ),
      ),
    );
});
(async () => {
  await Promise.all([loadCouncil(), refreshHistory()]);
  const saved = localStorage.getItem("council.activeChat");
  if (saved && historyItems.some((item) => item.id === saved)) {
    const chat = historyItems.find((item) => item.id === saved);
    await loadChat(chat.id, chat.temporary);
  } else renderConversation();
})();
