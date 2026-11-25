import { app } from "../../../scripts/app.js";

// Gibby Nodes - keep the preview widgets out of the queued prompt
// ---------------------------------------------------------------
// Core's DOM preview widgets (the mp4 player and the animated image host) are
// marked widget.serialize = false, which is the flag the saved-workflow
// serializer reads. graphToPrompt - the thing that builds the prompt the server
// executes - only skips a widget when widget.options.serialize === false, so the
// widget is queued as an extra input instead: "video-preview": "".
//
// A node's result-cache key is its own inputs plus every ancestor's, so that one
// phantom input invalidates the node and everything downstream of it whenever the
// widget is added or lost - which is on the first run that shows a video preview,
// and again after every page reload, since the widget is never restored from the
// saved workflow. Any node that previews a video pays for it.
//
// Pass the flag in the form graphToPrompt honours. The widgets carry no data
// (their value getter is always ""), so nothing is lost by not queueing them.
//
// TODO(frontend): delete this file once the frontend's addVideoDomWidget passes
// serialize:false in its options. Re-check after any frontend update by queueing
// a workflow and looking for a "video-preview" entry in the /prompt request body
// of a node that previews a video. Verified still needed on frontend 1.54.8.
const PREVIEW_WIDGETS = new Set(["video-preview", "$$comfy_animation_preview"]);

app.registerExtension({
    name: "gibby.previewWidgetsNotQueued",
    setup() {
        const proto = window.LGraphNode?.prototype;
        if (!proto?.addDOMWidget) return;
        const addDOMWidget = proto.addDOMWidget;
        proto.addDOMWidget = function (name, type, element, options = {}) {
            return addDOMWidget.call(
                this, name, type, element,
                PREVIEW_WIDGETS.has(name) ? { ...options, serialize: false } : options);
        };
    },
});
