(function () {
  "use strict";

  if (window.htmx) {
    window.htmx.config.includeIndicatorStyles = false;
  }

  const STATUS_REQUEST_ERROR = "Could not refresh status. Re-authenticate or reload.";
  const LOOKUP_REQUEST_ERROR = "Could not load recommendations. Re-authenticate or reload.";

  class TimeAgoController extends Stimulus.Controller {
    connect() {
      this._render();
      this._interval = window.setInterval(() => this._render(), 30000);
    }

    disconnect() {
      window.clearInterval(this._interval);
    }

    _render() {
      const iso = this.element.dataset.timeAgoValue;
      if (!iso) return;
      const then = new Date(iso);
      if (Number.isNaN(then.getTime())) return;

      if (this.element.tagName === "TIME") {
        this.element.dateTime = iso;
      }
      const seconds = Math.max(0, Math.round((Date.now() - then.getTime()) / 1000));
      this.element.textContent = `${this._relative(seconds)} (${iso})`;
    }

    _relative(seconds) {
      const units = [
        ["day", 86400],
        ["hour", 3600],
        ["minute", 60],
      ];
      for (const [label, secondsPerUnit] of units) {
        const value = Math.floor(seconds / secondsPerUnit);
        if (value >= 1) return `${value} ${label}${value === 1 ? "" : "s"} ago`;
      }
      return "just now";
    }
  }

  const application = Stimulus.Application.start();
  application.register("time-ago", TimeAgoController);

  let paused = false;
  let reducedMotionPause = false;
  let lastStatusKey = "";

  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function setBusy(el, busy) {
    if (!el) return;
    if (busy) el.setAttribute("aria-busy", "true");
    else el.removeAttribute("aria-busy");
  }

  function setAlert(el, message) {
    if (!el) return;
    if (message) {
      el.hidden = false;
      el.textContent = message;
    } else {
      el.hidden = true;
      el.textContent = "";
    }
  }

  function pollIntervalSeconds() {
    const status = qs("#status");
    return (status && status.getAttribute("data-refresh-interval")) || "30";
  }

  function syncPauseHint() {
    const hint = qs("[data-poll-hint]");
    if (!hint) return;
    if (paused && reducedMotionPause) {
      hint.hidden = false;
      hint.textContent = "Updates paused (reduced motion).";
    } else {
      hint.hidden = true;
    }
  }

  function setPaused(next) {
    paused = next;
    if (!next) reducedMotionPause = false;
    const status = qs("#status");
    const pauseBtn = qs("[data-status-pause]");
    if (status && window.htmx) {
      const interval = pollIntervalSeconds();
      status.setAttribute("hx-trigger", paused ? "refresh" : "refresh, every " + interval + "s");
      window.htmx.process(status);
    }
    if (pauseBtn) {
      pauseBtn.setAttribute("aria-pressed", paused ? "true" : "false");
      pauseBtn.textContent = paused ? "Resume updates" : "Pause updates";
    }
    syncPauseHint();
  }

  function currentUserId() {
    const input = qs("#user-id");
    if (input && input.value.trim()) return input.value.trim();
    const params = new URLSearchParams(window.location.search);
    return (params.get("user_id") || "").trim();
  }

  function syncDocumentTitle() {
    const parts = [];
    const card = qs("[data-latest-run]");
    if (card) {
      const status = card.getAttribute("data-run-status") || "";
      const stale = card.getAttribute("data-run-stale") === "true";
      if (status === "failed") parts.push("failed");
      else if (stale) parts.push("stale");
    }
    const userId = currentUserId();
    if (userId) parts.push(userId);
    parts.push("Cicerone dashboard");
    document.title = parts.join(" · ");
  }

  function statusKey(root) {
    if (!root) return "";
    if (root.querySelector("[data-latest-run-empty]")) return "empty";
    const card = root.querySelector("[data-latest-run]");
    if (!card) return "";
    const stale = card.getAttribute("data-run-stale") === "true" ? "stale" : "ok";
    return (card.getAttribute("data-run-status") || "unknown") + ":" + stale;
  }

  function statusAnnouncement(root) {
    if (root.querySelector("[data-latest-run-empty]")) return "No job runs recorded yet.";
    const card = root.querySelector("[data-latest-run]");
    if (!card) return "";
    const status = card.getAttribute("data-run-status") || "unknown";
    const stale = card.getAttribute("data-run-stale") === "true";
    if (status === "failed") return "Latest run failed.";
    if (stale) return "Latest run stale.";
    if (status === "success") return "Latest run succeeded.";
    return "Latest run " + status + ".";
  }

  function rememberStatus(root) {
    lastStatusKey = statusKey(root);
  }

  function announceStatusIfChanged(root) {
    const announcer = qs("#status-announcer");
    if (!announcer || !root) return;
    const key = statusKey(root);
    if (key === lastStatusKey) return;
    lastStatusKey = key;
    announcer.textContent = statusAnnouncement(root);
  }

  function lookupAnnouncement(root) {
    const alert = root.querySelector('[role="alert"]');
    if (alert) return (alert.textContent || "").trim();
    const summary = root.querySelector("[data-lookup-summary]");
    if (summary) {
      const recs = Number(summary.getAttribute("data-recs") || "0");
      const events = Number(summary.getAttribute("data-events") || "0");
      const user = (summary.getAttribute("data-user-id") || "").trim();
      let msg =
        recs +
        " recommendation" +
        (recs === 1 ? "" : "s") +
        " and " +
        events +
        " event" +
        (events === 1 ? "" : "s") +
        " for " +
        user;
      if (summary.getAttribute("data-fallback") === "true") {
        msg += ", using cold-start fallback";
      }
      return msg;
    }
    const status = root.querySelector('[role="status"]');
    if (status) return (status.textContent || "").trim();
    return "";
  }

  function announceLookup(root) {
    const announcer = qs("#recommendation-announcer");
    if (!announcer || !root) return;
    const text = lookupAnnouncement(root);
    if (text) announcer.textContent = text;
  }

  function requestTarget(detail) {
    const elt = detail.elt;
    if (elt && elt.id === "status") return qs("#status");
    if (elt instanceof HTMLFormElement && elt.hasAttribute("data-lookup-form")) {
      return qs("#recommendation-results");
    }
    return null;
  }

  function requestErrorMessage(elt) {
    if (elt && elt.id === "status") return STATUS_REQUEST_ERROR;
    if (elt instanceof HTMLFormElement && elt.hasAttribute("data-lookup-form")) {
      return LOOKUP_REQUEST_ERROR;
    }
    return "";
  }

  function handleRequestFailure(event) {
    const elt = event.detail && event.detail.elt;
    const msg = requestErrorMessage(elt);
    if (!msg) return;
    if (elt && elt.id === "status") setAlert(qs("[data-status-error]"), msg);
    else setAlert(qs("[data-lookup-error]"), msg);
  }

  function initDashboard() {
    document.body.addEventListener("htmx:beforeRequest", function (event) {
      setBusy(requestTarget(event.detail || {}), true);
    });

    document.body.addEventListener("htmx:afterRequest", function (event) {
      const detail = event.detail || {};
      setBusy(requestTarget(detail), false);
      const form = detail.elt || event.target;
      if (!(form instanceof HTMLFormElement) || !form.hasAttribute("data-lookup-form")) return;
      if (!detail.successful) return;
      const userId = new FormData(form).get("user_id");
      const url = new URL(window.location.href);
      if (typeof userId === "string" && userId.trim() !== "") {
        url.searchParams.set("user_id", userId.trim());
      } else {
        url.searchParams.delete("user_id");
      }
      window.history.replaceState(null, "", url.toString());
      syncDocumentTitle();
    });

    document.body.addEventListener("htmx:afterSwap", function (event) {
      const target = event.detail && event.detail.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.id === "status") {
        announceStatusIfChanged(target);
        syncDocumentTitle();
        setAlert(qs("[data-status-error]"), "");
      }
      if (target.id === "recommendation-results") {
        announceLookup(target);
        setAlert(qs("[data-lookup-error]"), "");
      }
    });

    document.body.addEventListener("htmx:sendError", handleRequestFailure);
    document.body.addEventListener("htmx:responseError", handleRequestFailure);

    const pauseBtn = qs("[data-status-pause]");
    const refreshBtn = qs("[data-status-refresh]");
    if (pauseBtn) {
      pauseBtn.addEventListener("click", function () {
        setPaused(!paused);
      });
    }
    if (refreshBtn) {
      refreshBtn.addEventListener("click", function () {
        const status = qs("#status");
        if (status && window.htmx) window.htmx.trigger(status, "refresh");
      });
    }
    rememberStatus(qs("#status"));
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      reducedMotionPause = true;
      setPaused(true);
    } else {
      // Markup is refresh-only so htmx cannot start a poll before this.
      setPaused(false);
      const status = qs("#status");
      if (status && window.htmx) window.htmx.trigger(status, "refresh");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDashboard);
  } else {
    initDashboard();
  }
})();
