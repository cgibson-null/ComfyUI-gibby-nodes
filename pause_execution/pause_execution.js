import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "GibbyNodes.PauseExecution",

    async setup() {
        // Every manual queue gets a fresh run id so the pause nodes block
        // again; the requeue after a Continue click (flagged below) keeps
        // its _continue run id and passes through them.
        const origQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function(e, t) {
            if (!window.gibby_is_auto_requeue) {
                const newRunId = Date.now().toString() + "_manual";

                const nodes = app.graph.computeExecutionOrder(false);
                for (const node of nodes) {
                    if (node.comfyClass === "GibbyPauseExecution") {
                        const runIdWidget = node.widgets?.find(w => w.name === "run_id");
                        if (runIdWidget) runIdWidget.value = newRunId;

                        if (node.debugDisplay) {
                            node.debugDisplay.textContent = "Run ID: " + newRunId;
                        }
                        if (node.timerDisplay) {
                            node.timerDisplay.textContent = "Ready to use";
                            node.timerDisplay.style.color = "#aaaaaa";
                        }
                        if (node.btnContinue) {
                            node.btnContinue.disabled = true;
                            node.btnContinue.style.opacity = "0.5";
                            node.btnContinue.style.cursor = "default";
                        }
                        if (node.btnStop) {
                            node.btnStop.disabled = true;
                            node.btnStop.style.opacity = "0.5";
                            node.btnStop.style.cursor = "default";
                        }
                    }
                }
            } else {
                window.gibby_is_auto_requeue = false;
            }
            return origQueuePrompt.apply(this, arguments);
        };

        api.addEventListener("executing", (event) => {
            const nodeId = event.detail;
            if (!nodeId) return;

            const node = app.graph.getNodeById(nodeId);
            if (node && node.comfyClass === "GibbyPauseExecution") {
                const blockWidget = node.widgets?.find(w => w.name === "pause_flow");
                const runIdWidget = node.widgets?.find(w => w.name === "run_id");

                if ((blockWidget && blockWidget.value === false) || (runIdWidget && runIdWidget.value.endsWith("_continue"))) {
                    if (node.timerDisplay) {
                        node.timerDisplay.textContent = "Transit completed!";
                        node.timerDisplay.style.color = "#00ff00";
                    }
                    return;
                }

                if (node.btnContinue) {
                    node.btnContinue.disabled = false;
                    node.btnContinue.style.opacity = "1";
                    node.btnContinue.style.cursor = "pointer";
                }
                if (node.btnStop) {
                    node.btnStop.disabled = false;
                    node.btnStop.style.opacity = "1";
                    node.btnStop.style.cursor = "pointer";
                }

                const emitSoundWidget = node.widgets?.find(w => w.name === "emit_sound");
                if (emitSoundWidget && emitSoundWidget.value) {
                    const audio = new Audio("/gibby/pause/ding.mp3");
                    audio.play().catch(e => console.warn("Audio:", e));
                }

                const timeoutWidget = node.widgets?.find(w => w.name === "timeout_seconds");
                let timeLeft = timeoutWidget ? timeoutWidget.value : 60;

                const onTimeoutWidget = node.widgets?.find(w => w.name === "on_timeout");
                const timeoutAction = onTimeoutWidget ? onTimeoutWidget.value : "continue";

                if (node.timerDisplay) {
                    node.timerDisplay.textContent = `Waiting for changes... ${timeLeft}s`;
                    node.timerDisplay.style.color = "#ffaa00";

                    if (node.timerInterval) clearInterval(node.timerInterval);

                    node.timerInterval = setInterval(() => {
                        timeLeft--;
                        if (timeLeft > 0) {
                            node.timerDisplay.textContent = `Waiting for changes... ${timeLeft}s`;
                        } else {
                            clearInterval(node.timerInterval);
                            node.timerDisplay.textContent = timeoutAction === "continue" ? "Continuing..." : "Expired (Stop)";
                            node.timerDisplay.style.color = timeoutAction === "continue" ? "#00ff00" : "#ff0000";

                            if (node.btnContinue) { node.btnContinue.disabled = true; node.btnContinue.style.opacity = "0.5"; node.btnContinue.style.cursor = "default"; }
                            if (node.btnStop) { node.btnStop.disabled = true; node.btnStop.style.opacity = "0.5"; node.btnStop.style.cursor = "default"; }
                        }
                    }, 1000);
                }
            }
        });
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GibbyPauseExecution") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            // The run_id widget is managed by this extension, hide it from the node.
            const obliterateGhostWidget = () => {
                const runIdWidget = this.widgets?.find(w => w.name === "run_id");
                if (runIdWidget) {
                    runIdWidget.type = "gibby_invisible";
                    runIdWidget.computeSize = () => [0, 0];
                    runIdWidget.draw = () => {};
                    runIdWidget.mouse = () => false;
                    runIdWidget.hidden = true;
                }
                if (this.inputs) {
                    const slotIdx = this.inputs.findIndex(i => i.name === "run_id");
                    if (slotIdx > -1) {
                        this.removeInput(slotIdx);
                    }
                }
            };

            obliterateGhostWidget();
            setTimeout(obliterateGhostWidget, 10);
            setTimeout(obliterateGhostWidget, 100);

            const container = document.createElement("div");
            container.style.cssText = "display:flex; flex-direction:column; gap:8px; width:100%; padding: 5px; box-sizing: border-box;";

            const timerDisplay = document.createElement("div");
            timerDisplay.textContent = "Ready to use";
            timerDisplay.style.cssText = "color: #aaaaaa; text-align: center; font-family: monospace; font-size: 14px; font-weight: bold;";

            const btnContinue = document.createElement("button");
            btnContinue.textContent = "Continue";
            btnContinue.disabled = true;
            btnContinue.style.cssText = "padding:6px; background:#2d6a2d; color:#fff; border:none; border-radius:4px; font-weight: bold; opacity: 0.5; cursor: default;";

            const btnStop = document.createElement("button");
            btnStop.textContent = "Stop Workflow";
            btnStop.disabled = true;
            btnStop.style.cssText = "padding:6px; background:#8b0000; color:#fff; border:none; border-radius:4px; font-weight: bold; opacity: 0.5; cursor: default;";

            const debugDisplay = document.createElement("div");
            debugDisplay.textContent = "Run ID: waiting...";
            debugDisplay.style.cssText = "color: #555555; text-align: center; font-family: monospace; font-size: 11px; margin-top: 4px; user-select: none; pointer-events: none;";

            container.appendChild(timerDisplay);
            container.appendChild(btnContinue);
            container.appendChild(btnStop);
            container.appendChild(debugDisplay);

            this.timerDisplay = timerDisplay;
            this.debugDisplay = debugDisplay;
            this.timerInterval = null;
            this.btnContinue = btnContinue;
            this.btnStop = btnStop;

            const sendAction = async (action) => {
                if (btnContinue.disabled) return;

                try {
                    if (this.timerInterval) clearInterval(this.timerInterval);
                    btnContinue.disabled = true;
                    btnStop.disabled = true;
                    btnContinue.style.opacity = "0.5";
                    btnStop.style.opacity = "0.5";
                    btnContinue.style.cursor = "default";
                    btnStop.style.cursor = "default";

                    if (action === "requeue") {
                        this.timerDisplay.textContent = "Saving and restarting...";
                        this.timerDisplay.style.color = "#00ff00";

                        window.gibby_is_auto_requeue = true;

                        const runIdWidget = this.widgets?.find(w => w.name === "run_id");
                        if (runIdWidget) {
                            const newId = Date.now().toString() + "_continue";
                            runIdWidget.value = newId;
                            debugDisplay.textContent = "Run ID: " + newId;
                        }

                        await app.queuePrompt();

                        await api.fetchApi("/gibby/pause/action", {
                            method: "POST",
                            body: JSON.stringify({ node_id: this.id.toString(), action: "stop" }),
                        });

                    } else {
                        await api.fetchApi("/gibby/pause/action", {
                            method: "POST",
                            body: JSON.stringify({ node_id: this.id.toString(), action: "stop" }),
                        });
                        this.timerDisplay.textContent = "Permanently stopped";
                        this.timerDisplay.style.color = "#ff0000";
                    }

                } catch (e) {
                    console.error("API error:", e);
                }
            };

            btnContinue.addEventListener("click", () => sendAction("requeue"));
            btnStop.addEventListener("click", () => sendAction("stop"));

            this.addDOMWidget("gibby_pause_ui", "GIBBY_PAUSE_UI", container);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const runIdWidget = this.widgets?.find(w => w.name === "run_id");
            if (runIdWidget) {
                runIdWidget.type = "gibby_invisible";
                runIdWidget.computeSize = () => [0, 0];
                runIdWidget.draw = () => {};
                runIdWidget.mouse = () => false;
                runIdWidget.hidden = true;
            }
            if (this.inputs) {
                const slotIdx = this.inputs.findIndex(i => i.name === "run_id");
                if (slotIdx > -1) {
                    this.removeInput(slotIdx);
                }
            }
        };
    },
});
