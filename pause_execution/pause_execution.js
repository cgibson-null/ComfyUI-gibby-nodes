import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "GibbyNodes.PauseExecution",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GibbyPauseExecution") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            const btn = document.createElement("button");
            btn.textContent = "Continue";
            btn.style.cssText = "width:100%;padding:4px 12px;background:#2d6a2d;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;box-sizing:border-box;";
            btn.addEventListener("click", () => {
                const stopWidget = this.widgets?.find(w => w.name === "stop");
                if (!stopWidget) return;
                stopWidget.value = false;
                app.queuePrompt().then(() => {
                    setTimeout(() => { stopWidget.value = true; }, 2000);
                });
            });

            this.addDOMWidget("pause_continue", "GIBBY_PAUSE_CONTINUE", btn, {
                getHeight: () => 28,
            });
        };
    },
});
