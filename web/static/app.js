(function () {
  "use strict";

  var grid = document.getElementById("grid");
  var emptyState = document.getElementById("empty_state");
  var addForm = document.getElementById("add_form");
  var toggleAddBtn = document.getElementById("toggle_add_btn");
  var doAddBtn = document.getElementById("do_add_btn");
  var cancelAddBtn = document.getElementById("cancel_add_btn");
  var inpUrl = document.getElementById("inp_url");
  var inpName = document.getElementById("inp_name");

  var vitDot = document.getElementById("vit_dot");
  var vitLabel = document.getElementById("vit_label");
  var vlmDot = document.getElementById("vlm_dot");
  var vlmLabel = document.getElementById("vlm_label");

  var modalOverlay = document.getElementById("modal_overlay");
  var modalTitle = document.getElementById("modal_title");
  var modalClose = document.getElementById("modal_close");
  var modalCancel = document.getElementById("modal_cancel");
  var modalSave = document.getElementById("modal_save");
  var tabBtns = document.querySelectorAll(".modal-tabs button");

  var rngYolo = document.getElementById("rng_yolo");
  var valYolo = document.getElementById("val_yolo");
  var rngVit = document.getElementById("rng_vit");
  var valVit = document.getElementById("val_vit");
  var chkAgent = document.getElementById("chk_agent");
  var inpVlmInterval = document.getElementById("inp_vlm_interval");

  var roiImg = document.getElementById("roi_img");
  var roiCanvas = document.getElementById("roi_canvas");
  var roiCtx = roiCanvas.getContext("2d");
  var roiUndoBtn = document.getElementById("roi_undo_btn");
  var roiFinishBtn = document.getElementById("roi_finish_btn");
  var roiClearBtn = document.getElementById("roi_clear_btn");
  var roiListEl = document.getElementById("roi_list");

  var roiClassTags = document.getElementById("roi_class_tags");
  var globalClassTags = document.getElementById("global_class_tags");

  var ROI_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4", "#f97316", "#ec4899"];

  var cardMap = {};
  var allClasses = [];
  var editStreamId = null;
  var editRois = [];
  var currentPoints = [];
  var selectedRoiClasses = [];
  var selectedGlobalClasses = [];

  // ── API helpers ──

  function api(method, path, body) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (r) { return r.json(); });
  }

  // ── Add form ──

  toggleAddBtn.onclick = function () {
    addForm.classList.toggle("open");
  };
  cancelAddBtn.onclick = function () {
    addForm.classList.remove("open");
  };
  doAddBtn.onclick = function () {
    var url = (inpUrl.value || "").trim();
    if (!url) { alert("请输入 RTSP 地址"); return; }
    api("POST", "/api/streams", {
      rtsp_url: url,
      name: (inpName.value || "").trim() || undefined,
      enabled: true
    }).then(function (r) {
      if (r.error) { alert(r.error); return; }
      inpUrl.value = "";
      inpName.value = "";
      addForm.classList.remove("open");
      loadCards();
    });
  };

  // ── Cards ──

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function badgeFor(st, enabled) {
    if (!enabled) return '<span class="badge badge-stopped">已停止</span>';
    if (!st || !st.vit_result) return '<span class="badge badge-waiting">等待中</span>';
    var lbl = ((st.vit_result.pred_label || "") + "").trim().toLowerCase();
    if (lbl === "normal") return '<span class="badge badge-normal">正常</span>';
    return '<span class="badge badge-anomaly">异常</span>';
  }

  function statusHtml(st, vitLoaded) {
    if (!st) return '<span class="vit-line">--</span>';
    if (st.error_message) return '<span class="error-line">错误: ' + esc(st.error_message) + '</span>';
    if (!vitLoaded) {
      return '<span class="vit-line">ViT 未加载</span>';
    }
    if (st.vit_result) {
      var lbl = ((st.vit_result.pred_label || "") + "").trim().toLowerCase();
      var prob = st.vit_result.pred_prob != null ? Number(st.vit_result.pred_prob).toFixed(2) : "--";
      if (lbl === "normal") {
        return '<span class="vit-line">ViT 检测：正常 (' + prob + ')</span>';
      }
      return '<span class="vit-line" style="color:var(--danger)">ViT 检测：异常 (' + prob + ')</span>';
    }
    return '<span class="vit-line">ViT 检测：等待中...</span>';
  }

  function vlmTitle(vlmFinetuned) {
    return vlmFinetuned ? "专业 AI 模型解释" : "AI 大模型解释";
  }

  function updateVlmBox(card, st, vlmLoaded, vlmFinetuned) {
    var statusEl = card.querySelector(".card-status");
    if (!statusEl) return;
    var box = statusEl.querySelector(".vlm-box");
    var hasResult = vlmLoaded && st && st.agent_result && (st.agent_result.reasoning || "").trim();
    if (!hasResult) {
      if (box) box.style.display = "none";
      return;
    }
    var a = st.agent_result;
    var reason = (a.reasoning || "").trim() || "(无解释文本)";
    // 如果后端标记了解析错误，给出更明显的提示
    if (a.error) {
      reason = "⚠️ " + reason;
    }
    if (reason.length > 200) { reason = reason.substring(0, 200) + "…"; }
    var titleText = vlmTitle(vlmFinetuned);
    if (!box) {
      box = document.createElement("div");
      box.className = "vlm-box";
      box.innerHTML =
        '<div class="vlm-head">' +
          '<span class="vlm-title">' + esc(titleText) + '</span>' +
          '<span class="vlm-arrow">▼</span>' +
        '</div>' +
        '<div class="vlm-body"><div class="vlm-body-inner"></div></div>';
      box.querySelector(".vlm-head").onclick = function () { box.classList.toggle("open"); };
      statusEl.appendChild(box);
    }
    box.style.display = "";
    var titleEl = box.querySelector(".vlm-title");
    if (titleEl && titleEl.textContent !== titleText) titleEl.textContent = titleText;
    var prev = box.dataset.reason || "";
    if (reason !== prev) {
      box.querySelector(".vlm-body-inner").textContent = reason;
      box.dataset.reason = reason;
      box.classList.remove("flash");
      void box.offsetWidth;
      box.classList.add("flash");
    }
  }

  function renderCard(stream) {
    var id = stream.id;
    var enabled = stream.enabled !== false;
    var card = document.createElement("div");
    card.className = "card";
    card.setAttribute("data-id", id);

    card.innerHTML =
      '<div class="card-header">' +
        '<span class="name">' + esc(stream.name || id) + '</span>' +
        '<span class="badge-slot">' + badgeFor(null, enabled) + '</span>' +
      '</div>' +
      '<div class="card-video">' +
        (enabled ? '<img src="/video/' + encodeURIComponent(id) + '" alt="video">' : '<div style="height:240px;display:flex;align-items:center;justify-content:center;color:var(--text3)">已停止</div>') +
      '</div>' +
      '<div class="card-status">--</div>' +
      '<div class="card-actions">' +
        '<button class="btn btn-sm btn-outline settings-btn">⚙️ 设置</button>' +
        (enabled
          ? '<button class="btn btn-sm btn-danger stop-btn">⏹ 停止</button>'
          : '<button class="btn btn-sm btn-success start-btn">▶ 启动</button>') +
        '<button class="btn btn-sm btn-outline del-btn" style="margin-left:auto">🗑 删除</button>' +
      '</div>';

    var settingsBtn = card.querySelector(".settings-btn");
    var stopBtn = card.querySelector(".stop-btn");
    var startBtn = card.querySelector(".start-btn");
    var delBtn = card.querySelector(".del-btn");

    settingsBtn.onclick = function () { openSettings(id); };
    if (stopBtn) stopBtn.onclick = function () { api("POST", "/api/streams/" + id + "/stop").then(loadCards); };
    if (startBtn) startBtn.onclick = function () { api("POST", "/api/streams/" + id + "/start").then(loadCards); };
    delBtn.onclick = function () {
      if (confirm("确定删除该视频流？")) api("DELETE", "/api/streams/" + id).then(loadCards);
    };

    return card;
  }

  function loadCards() {
    api("GET", "/api/streams").then(function (data) {
      var streams = data.streams || [];
      updateModelStatus(data.vit_loaded, data.vlm_loaded, data.vlm_finetuned, data.vlm_base);
      grid.innerHTML = "";
      cardMap = {};
      if (streams.length === 0) {
        emptyState.style.display = "";
        return;
      }
      emptyState.style.display = "none";
      streams.forEach(function (s) {
        var card = renderCard(s);
        grid.appendChild(card);
        cardMap[s.id] = { card: card, stream: s };
      });
    });
  }

  function updateModelStatus(vit, vlm, vlmFinetuned, vlmBase) {
    vitDot.className = "status-dot " + (vit ? "on" : "off");
    vitLabel.textContent = "ViT: " + (vit ? "已加载" : "未加载");
    vlmDot.className = "status-dot " + (vlm ? "on" : "off");
    if (!vlm) {
      vlmLabel.textContent = "大模型: 未加载";
    } else if (vlmBase) {
      vlmLabel.textContent = "⚠️ 基础模型(不可用)";
    } else if (vlmFinetuned) {
      vlmLabel.textContent = "专业 AI 模型";
    } else {
      vlmLabel.textContent = "大模型";
    }
  }

  // ── Polling ──

  var _pollLock = {};
  function pollStatus() {
    Object.keys(cardMap).forEach(function (id) {
      if (_pollLock[id]) return;
      _pollLock[id] = true;
      api("GET", "/api/streams/" + id + "/status").then(function (res) {
        _pollLock[id] = false;
        var entry = cardMap[id];
        if (!entry) return;
        var card = entry.card;
        var st = res.status || {};
        var enabled = res.enabled !== false;

        var badgeSlot = card.querySelector(".badge-slot");
        if (badgeSlot) badgeSlot.innerHTML = badgeFor(st, enabled);

        // Update only the ViT line; VLM box is managed independently to preserve open/close state
        var statusEl = card.querySelector(".card-status");
        if (statusEl) {
          var vitLine = statusEl.querySelector(".vit-line, .error-line");
          var html = statusHtml(st, res.vit_loaded);
          if (vitLine) {
            // replace just the ViT line, keep vlm-box if any
            var tmp = document.createElement("div");
            tmp.innerHTML = html;
            statusEl.replaceChild(tmp.firstChild, vitLine);
          } else {
            // first time: clear "--" placeholder and inject ViT line first
            statusEl.innerHTML = html;
          }
        }

        updateVlmBox(card, st, res.vlm_loaded, res.vlm_finetuned);

        updateModelStatus(res.vit_loaded, res.vlm_loaded, res.vlm_finetuned, res.vlm_base);
      }).catch(function () { _pollLock[id] = false; });
    });
  }

  // ── Modal tabs ──

  tabBtns.forEach(function (btn) {
    btn.onclick = function () {
      tabBtns.forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      ["tab_basic", "tab_roi", "tab_alarm"].forEach(function (t) {
        document.getElementById(t).style.display = "none";
      });
      document.getElementById(btn.getAttribute("data-tab")).style.display = "";
    };
  });

  modalClose.onclick = closeModal;
  modalCancel.onclick = closeModal;
  modalOverlay.onclick = function (e) { if (e.target === modalOverlay) closeModal(); };

  function closeModal() {
    modalOverlay.classList.remove("open");
    editStreamId = null;
  }

  // ── Settings modal ──

  function openSettings(streamId) {
    editStreamId = streamId;
    api("GET", "/api/streams/" + streamId + "/status").then(function (res) {
      modalTitle.textContent = (res.name || streamId) + " - 设置";
      rngYolo.value = res.yolo_confidence || 0.5;
      valYolo.textContent = Number(rngYolo.value).toFixed(2);
      rngVit.value = res.vit_threshold || 0.6;
      valVit.textContent = Number(rngVit.value).toFixed(2);
      chkAgent.checked = res.agent_enabled !== false;
      var vlmInt = res.vlm_auto_interval_sec;
      inpVlmInterval.value = (vlmInt == null) ? 16 : vlmInt;

      editRois = (res.rois || []).map(function (poly) { return poly.slice(); });
      currentPoints = [];
      selectedRoiClasses = (res.roi_alarm_classes || ["person"]).slice();
      selectedGlobalClasses = (res.global_alarm_classes || ["fire", "smoke"]).slice();

      tabBtns[0].click();
      modalOverlay.classList.add("open");

      loadSnapshot(streamId);
      renderClassPickers();
    });
  }

  rngYolo.oninput = function () { valYolo.textContent = Number(rngYolo.value).toFixed(2); };
  rngVit.oninput = function () { valVit.textContent = Number(rngVit.value).toFixed(2); };

  modalSave.onclick = function () {
    if (!editStreamId) return;
    var vlmIntervalVal = parseFloat(inpVlmInterval.value);
    if (isNaN(vlmIntervalVal) || vlmIntervalVal < 0) vlmIntervalVal = 0;
    var updates = {
      yolo_confidence: parseFloat(rngYolo.value),
      vit_threshold: parseFloat(rngVit.value),
      agent_enabled: chkAgent.checked,
      rois: editRois,
      roi_alarm_classes: selectedRoiClasses,
      global_alarm_classes: selectedGlobalClasses,
      vlm_auto_interval_sec: vlmIntervalVal
    };
    api("PATCH", "/api/streams/" + editStreamId, updates).then(function () {
      closeModal();
      loadCards();
    });
  };

  // ── ROI drawing ──

  function loadSnapshot(streamId) {
    roiImg.onload = function () {
      roiCanvas.width = roiImg.naturalWidth;
      roiCanvas.height = roiImg.naturalHeight;
      roiCanvas.style.width = roiImg.clientWidth + "px";
      roiCanvas.style.height = roiImg.clientHeight + "px";
      drawAllRois();
    };
    roiImg.src = "/api/streams/" + streamId + "/snapshot?t=" + Date.now();
  }

  function getCanvasPos(e) {
    var rect = roiCanvas.getBoundingClientRect();
    var scaleX = roiCanvas.width / rect.width;
    var scaleY = roiCanvas.height / rect.height;
    return [
      Math.round((e.clientX - rect.left) * scaleX),
      Math.round((e.clientY - rect.top) * scaleY)
    ];
  }

  roiCanvas.onclick = function (e) {
    var pt = getCanvasPos(e);
    currentPoints.push(pt);
    drawAllRois();
  };

  roiCanvas.ondblclick = function (e) {
    e.preventDefault();
    finishCurrentRoi();
  };

  roiUndoBtn.onclick = function () {
    currentPoints.pop();
    drawAllRois();
  };

  roiFinishBtn.onclick = finishCurrentRoi;

  roiClearBtn.onclick = function () {
    editRois = [];
    currentPoints = [];
    drawAllRois();
    renderRoiList();
  };

  function finishCurrentRoi() {
    if (currentPoints.length < 3) {
      alert("禁区至少需要 3 个点");
      return;
    }
    editRois.push(currentPoints.slice());
    currentPoints = [];
    drawAllRois();
    renderRoiList();
  }

  function drawAllRois() {
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    editRois.forEach(function (poly, i) {
      drawPoly(poly, ROI_COLORS[i % ROI_COLORS.length], true);
    });
    if (currentPoints.length > 0) {
      drawPoly(currentPoints, ROI_COLORS[editRois.length % ROI_COLORS.length], false);
    }
  }

  function drawPoly(pts, color, closed) {
    if (pts.length === 0) return;
    roiCtx.beginPath();
    roiCtx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) {
      roiCtx.lineTo(pts[i][0], pts[i][1]);
    }
    if (closed) roiCtx.closePath();
    roiCtx.strokeStyle = color;
    roiCtx.lineWidth = 2;
    roiCtx.stroke();
    if (closed) {
      roiCtx.fillStyle = color.replace(")", ",0.15)").replace("rgb", "rgba").replace("#", "");
      try {
        var r = parseInt(color.slice(1, 3), 16);
        var g = parseInt(color.slice(3, 5), 16);
        var b = parseInt(color.slice(5, 7), 16);
        roiCtx.fillStyle = "rgba(" + r + "," + g + "," + b + ",0.15)";
        roiCtx.fill();
      } catch (e) {}
    }
    pts.forEach(function (pt) {
      roiCtx.beginPath();
      roiCtx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
      roiCtx.fillStyle = color;
      roiCtx.fill();
    });
  }

  function renderRoiList() {
    roiListEl.innerHTML = "";
    editRois.forEach(function (poly, i) {
      var div = document.createElement("div");
      div.className = "roi-item";
      div.innerHTML =
        '<span class="roi-color" style="background:' + ROI_COLORS[i % ROI_COLORS.length] + '"></span>' +
        '<span>禁区 ' + (i + 1) + ' (' + poly.length + ' 个点)</span>' +
        '<button class="btn btn-sm btn-danger" style="margin-left:auto">删除</button>';
      div.querySelector("button").onclick = function () {
        editRois.splice(i, 1);
        drawAllRois();
        renderRoiList();
      };
      roiListEl.appendChild(div);
    });
  }

  // ── Class pickers ──

  function loadClasses() {
    api("GET", "/api/classes").then(function (data) {
      allClasses = data.classes || [];
    });
  }

  function renderClassPickers() {
    renderTagGroup(roiClassTags, selectedRoiClasses, function (sel) { selectedRoiClasses = sel; });
    renderTagGroup(globalClassTags, selectedGlobalClasses, function (sel) { selectedGlobalClasses = sel; });
  }

  function renderTagGroup(container, selected, onUpdate) {
    container.innerHTML = "";
    allClasses.forEach(function (cls) {
      var tag = document.createElement("span");
      tag.className = "class-tag" + (selected.indexOf(cls) >= 0 ? " selected" : "");
      tag.textContent = cls;
      tag.onclick = function () {
        var idx = selected.indexOf(cls);
        if (idx >= 0) selected.splice(idx, 1);
        else selected.push(cls);
        onUpdate(selected);
        renderTagGroup(container, selected, onUpdate);
      };
      container.appendChild(tag);
    });
  }

  // ── Init ──

  loadClasses();
  loadCards();
  setInterval(pollStatus, 2000);
})();
