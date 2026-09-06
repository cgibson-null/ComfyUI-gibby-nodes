import { app } from "../../../scripts/app.js";

let _modelListsPromise = null;

function fetchModelLists() {
    const cacheBust = new Date().getTime();
    return fetch(`/object_info/Gibby_ContextLoader?t=${cacheBust}`)
        .then((r) => r.json())
        .then((data) => {
            const nodeDef = data?.Gibby_ContextLoader;
            if (!nodeDef) return null;
            const modeInput = nodeDef.input?.required?.mode;
            if (!modeInput || modeInput[0] !== "COMFY_DYNAMICCOMBO_V3") return null;
            return modeInput[1]?.options || null;
        })
        .catch(() => null);
}

function getModelLists() {
    if (!_modelListsPromise) {
        _modelListsPromise = fetchModelLists();
    }
    return _modelListsPromise;
}

function invalidateModelListsCache() {
    _modelListsPromise = null;
}

app.registerExtension({
    name: "Gibby.ContextLoaderRefresh",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Gibby_ContextLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this._refreshContextLoaderOptions = async () => {
                invalidateModelListsCache();
                const newOptions = await getModelLists();
                if (!newOptions) return;

                for (const modeOpt of newOptions) {
                    const inputs = modeOpt?.inputs?.required || {};
                    const optInputs = modeOpt?.inputs?.optional || {};
                    for (const [name, spec] of [...Object.entries(inputs), ...Object.entries(optInputs)]) {
                        if (!Array.isArray(spec)) continue;
                        if (spec[0] === "COMBO") {
                            const widget = this.widgets?.find((w) => w.name === `mode.${name}`);
                            if (widget?.options?.values) {
                                widget.options.values.splice(0, widget.options.values.length, ...spec[1].options);
                                widget._state?.options?.values?.splice(0, widget._state.options.values.length, ...spec[1].options);
                            }
                        } else if (spec[0] === "COMFY_DYNAMICCOMBO_V3") {
                            // Nested dynamic combo (e.g. clip_count)
                            for (const subOpt of spec[1]?.options || []) {
                                const subInputs = subOpt?.inputs?.required || {};
                                const subOptInputs = subOpt?.inputs?.optional || {};
                                for (const [subName, subSpec] of [...Object.entries(subInputs), ...Object.entries(subOptInputs)]) {
                                    if (!Array.isArray(subSpec) || subSpec[0] !== "COMBO") continue;
                                    const widget = this.widgets?.find((w) => w.name === `mode.${name}.${subName}`);
                                    if (widget?.options?.values) {
                                        widget.options.values.splice(0, widget.options.values.length, ...subSpec[1].options);
                                        widget._state?.options?.values?.splice(0, widget._state.options.values.length, ...subSpec[1].options);
                                    }
                                }
                            }
                        }
                    }
                }
            };

            return result;
        };

        nodeType.prototype.refreshComboInNode = function () {
            if (this._refreshContextLoaderOptions) {
                this._refreshContextLoaderOptions();
            }
        };
    },
});
