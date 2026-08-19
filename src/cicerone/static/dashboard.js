// Dashboard page bootstrap: starts the Stimulus application and registers
// the one controller this page needs so far (relative-time formatting for
// manifest timestamps -- the raw ISO string alone makes "is this stale?"
// harder to answer at a glance). htmx (vendor/htmx.min.js) needs no JS
// wiring here: its polling requests reuse the browser's cached HTTP Basic
// Auth credentials for this origin automatically.
(function () {
  "use strict";

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

  document.body.addEventListener("htmx:afterRequest", function (event) {
    const detail = event.detail || {};
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
    window.history.replaceState(null, "", `${url.pathname}${url.search}`);
  });
})();
