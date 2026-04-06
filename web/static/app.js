(function () {
  var grid = document.getElementById("grid");
  var rtspInput = document.getElementById("rtsp_url");
  var nameInput = document.getElementById("stream_name");
  var addBtn = document.getElementById("add_btn");

  function fetchStreams() {
    return fetch("/api/streams").then(function (r) { return r.json(); });
  }

  function fetchStatus(id) {
    return fetch("/api/streams/" + encodeURIComponent(id) + "/status").then(function (r) { return r.json(); });
  }

  function renderCard(stream) {
    var id = stream.id;
    var name = stream.name || id;
    var enabled = stream.enabled !== false;
    var statusEl = document.createElement("div");
    statusEl.className = "status";
    statusEl.setAttribute("data-stream-id", id);

    var actions = document.createElement("div");
    actions.className = "actions";
    var stopBtn = document.createElement("button");
    stopBtn.className = "stop";
    stopBtn.textContent = "停止";
    stopBtn.onclick = function () {
      fetch("/api/streams/" + encodeURIComponent(id) + "/stop", { method: "POST" })
        .then(function () { loadCards(); });
    };
    var startBtn = document.createElement("button");
    startBtn.className = "start";
    startBtn.textContent = "启动";
    startBtn.onclick = function () {
      fetch("/api/streams/" + encodeURIComponent(id) + "/start", { method: "POST" })
        .then(function () { loadCards(); });
    };
    var delBtn = document.createElement("button");
    delBtn.className = "del";
    delBtn.textContent = "删除";
    delBtn.onclick = function () {
      if (confirm("确定删除该流？")) {
        fetch("/api/streams/" + encodeURIComponent(id), { method: "DELETE" })
          .then(function () { loadCards(); });
      }
    };
    actions.appendChild(stopBtn);
    actions.appendChild(startBtn);
    actions.appendChild(delBtn);

    var wrap = document.createElement("div");
    wrap.className = "video-wrap";
    var img = document.createElement("img");
    img.src = "/video/" + encodeURIComponent(id);
    img.alt = name;
    wrap.appendChild(img);

    var card = document.createElement("div");
    card.className = "card";
    card.setAttribute("data-stream-id", id);
    var h3 = document.createElement("h3");
    h3.textContent = name + (enabled ? "" : " (已停止)");
    card.appendChild(h3);
    card.appendChild(wrap);
    card.appendChild(statusEl);
    card.appendChild(actions);
    return { card: card, statusEl: statusEl, startBtn: startBtn, stopBtn: stopBtn };
  }

  var cardMap = {};
  var vitLoaded = false;

  function loadCards() {
    fetchStreams().then(function (data) {
      var streams = data.streams || [];
      vitLoaded = !!data.vit_loaded;
      grid.innerHTML = "";
      cardMap = {};
      streams.forEach(function (stream) {
        var out = renderCard(stream);
        grid.appendChild(out.card);
        cardMap[stream.id] = out;
      });
    });
  }

  function pollStatus() {
    Object.keys(cardMap).forEach(function (id) {
      fetchStatus(id).then(function (res) {
        var statusEl = cardMap[id] && cardMap[id].statusEl;
        if (!statusEl) return;
        var st = res.status || {};
        var vLoaded = res.vit_loaded !== false;
        if (res.vit_loaded !== undefined) vitLoaded = res.vit_loaded;
        var htmlParts = [];
        if (st.error_message) {
          statusEl.className = "status error";
          var err = String(st.error_message).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          htmlParts.push("错误: " + err);
        } else {
          statusEl.className = "status";
          var monitorLine = "";
          if (!vLoaded) {
            monitorLine = "监控情况：ViT 未加载（需 checkpoint_best.pt）";
          } else if (st.vit_result) {
            var lbl = (st.vit_result.pred_label || "").toString().trim().toLowerCase();
            monitorLine = (lbl === "normal") ? "监控情况：正常" : "监控情况：异常";
          } else {
            monitorLine = "监控情况：等待中…";
          }
          htmlParts.push(monitorLine);
          var prob = st.vit_result && st.vit_result.pred_prob != null ? Number(st.vit_result.pred_prob) : 0;
          var isAnomaly = st.vit_result && (st.vit_result.pred_label || "").toString().trim().toLowerCase() !== "normal";
          if (isAnomaly && prob > 0.6 && st.agent_result) {
            var a = st.agent_result;
            var conf = a.confidence != null ? Number(a.confidence).toFixed(2) : "—";
            var reason = (a.reasoning && a.reasoning.trim()) ? String(a.reasoning).trim().slice(0, 120) : "";
            reason = reason.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
            htmlParts.push('<span class="status-agent-line">AI大模型：置信度：' + conf + "，情况：" + reason + "</span>");
          }
          if (isAnomaly && prob > 0.6) statusEl.classList.add("agent");
        }
        statusEl.innerHTML = htmlParts.length ? htmlParts.join("<br>") : "—";
      }).catch(function () {});
    });
  }

  addBtn.onclick = function () {
    var url = (rtspInput.value || "").trim();
    if (!url) {
      alert("请输入 RTSP 地址");
      return;
    }
    fetch("/api/streams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rtsp_url: url, name: (nameInput.value || "").trim() || undefined, enabled: true })
    }).then(function (r) {
      if (r.ok) { rtspInput.value = ""; nameInput.value = ""; loadCards(); }
      else r.json().then(function (e) { alert(e.error || "添加失败"); });
    });
  };

  loadCards();
  setInterval(pollStatus, 2000);
})();
