"""
Prompt travel syntax
--------------------
The a1111/Forge prompt-travel syntax carried in a context's prompt text, parsed
into a per-step prompt plan:

- ``[a:b]`` / ``[a:b|3]`` - alternate between a and b, every step or every 3.
- ``[a:b:3]`` / ``[a:b:.4]`` - a until step 3 (or 40% of the steps), b from there.
- ``[a:b:8:4]`` - crossfade: the two prompts are encoded separately and mixed.

Only square brackets travel. Round brackets are ComfyUI's weight syntax and stay
untouched, including their colons, so ``(a:1.5)`` never reads as a travel group.

Shared by KSampler (Context), which samples the plan step by step, and by the
context's own conditioning, which is built from the plan's first step so the
travel syntax is never encoded as literal text.
"""


def _find_matching(text, start, open_ch, close_ch):
    """The index of the closing char matching the opening char at start; -1 when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_travel(text):
    """Split a [...] group content on the top-level colons: colons inside () and [] stay in their part."""
    parts, cur, depth = [], [], 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == ":" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_travel_group(content):
    """The parsed contents of a [...] group:
    - [before:after:step] -> a (before, after, step) switch; before is active until
      the step, after from it on (step with a dot=fraction of steps, without=absolute).
    - [before:after:step:radius] -> a ("blend", before, after, step, radius) crossfade:
      the prompts mix linearly around the step, radius is how many steps each side
      stays pure (with a dot a fraction of the steps).
    - [before:after] or [before:after|period] -> an ("alt", before, after, period)
      alternate that flips between before and after; period is how many steps each
      side stays active (default 1, with a dot a fraction of the steps).
    Otherwise the content is plain text."""
    parts = _split_travel(content)
    if len(parts) == 2:
        after_text, period = parts[1], ""
        if "|" in after_text:
            after_text, _, period = after_text.partition("|")
            try:
                float(period)
            except ValueError:
                return content
        return ("alt", _parse_travel_expr(parts[0]), _parse_travel_expr(after_text), period)
    if len(parts) < 3:
        return content
    try:
        float(parts[-1])
    except ValueError:
        return content
    if len(parts) >= 4:
        try:
            float(parts[-2])
        except ValueError:
            pass
        else:
            return ("blend", _parse_travel_expr(":".join(parts[:-3])), _parse_travel_expr(parts[-3]),
                    parts[-2], parts[-1])
    return (_parse_travel_expr(":".join(parts[:-2])), _parse_travel_expr(parts[-2]), parts[-1])


def _parse_travel_expr(text):
    """Parse a travel prompt into a tree: plain text is a string, a sequence is a
    list, a [before:after:step] group a (before, after, step) switch, and a
    [before:after] group an ("alt", ...) alternate. An unbalanced bracket stays
    as-is, and () groups are atomic weight syntax."""
    nodes, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "([":
            j = _find_matching(text, i, ch, ")" if ch == "(" else "]")
            if j == -1:
                nodes.append(ch)
                i += 1
            elif ch == "(":
                nodes.append(text[i:j + 1])
                i = j + 1
            else:
                nodes.append(_parse_travel_group(text[i + 1:j]))
                i = j + 1
        else:
            j = i
            while j < n and text[j] not in "([":
                j += 1
            nodes.append(text[i:j])
            i = j
    out = []
    for node in nodes:
        if isinstance(node, str) and out and isinstance(out[-1], str):
            out[-1] += node
        else:
            out.append(node)
    return out[0] if len(out) == 1 else out


def _resolve_travel_step(node, steps_value):
    """Resolve every switch step, blend step/radius and alternate period of a travel
    tree against the total step count (a value with a dot is a fraction of the steps,
    without one an absolute count) and clamp it into range."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return [_resolve_travel_step(n, steps_value) for n in node]
    if node[0] == "alt":
        _, before, after, period = node
        if period:
            val = int(float(period) * steps_value) if "." in period else int(float(period))
            period = max(1, val)
        else:
            period = 1
        return ("alt", _resolve_travel_step(before, steps_value),
                _resolve_travel_step(after, steps_value), period)
    if node[0] == "blend":
        _, before, after, center, radius = node
        c = int(float(center) * steps_value) if "." in center else int(float(center))
        r = int(float(radius) * steps_value) if "." in radius else int(float(radius))
        return ("blend", _resolve_travel_step(before, steps_value),
                _resolve_travel_step(after, steps_value),
                max(0, min(steps_value, c)), max(1, r))
    before, after, step = node
    val = int(float(step) * steps_value) if "." in step else int(float(step))
    return (_resolve_travel_step(before, steps_value), _resolve_travel_step(after, steps_value),
            max(0, min(steps_value, val)))


def _pieces(item):
    """A rendered prompt piece as (text, weight) endpoints: a plain string is a
    single full-strength endpoint, a ("mix", ...) item its endpoints."""
    return [(item, 1.0)] if isinstance(item, str) else item[1]


def _mix_pieces(a, b, w):
    """Blend two prompt pieces at weight w of the second: the endpoints of both
    pieces, scaled by their side's weight."""
    return [(t, wt * (1.0 - w)) for t, wt in _pieces(a)] + [(t, wt * w) for t, wt in _pieces(b)]


def _fold_pieces(items):
    """Join a sequence of prompt pieces into one prompt: the endpoints are the
    cartesian product of the pieces' endpoints with multiplied weights; a single
    endpoint collapses back to a plain string."""
    out = [("", 1.0)]
    for item in items:
        out = [(t + t2, wt * wt2) for t, wt in out for t2, wt2 in _pieces(item)]
    return out[0][0] if len(out) == 1 else ("mix", out)


def _render_travel(node, step):
    """The prompt of a travel tree at a (global) step index: a string for plain and
    switching prompts, a ("mix", [(text, weight), ...]) item where a crossfade
    mixes the endpoints (step is the 1-based step number the weight is measured at)."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return _fold_pieces([_render_travel(n, step) for n in node])
    if node[0] == "alt":
        _, before, after, period = node
        return _render_travel(before if (step // period) % 2 == 0 else after, step)
    if node[0] == "blend":
        _, before, after, center, radius = node
        w = (step + 1 - center + radius) / (2.0 * radius)
        if w <= 0.0:
            return _render_travel(before, step)
        if w >= 1.0:
            return _render_travel(after, step)
        return ("mix", _mix_pieces(_render_travel(before, step), _render_travel(after, step), w))
    before, after, start = node
    return _render_travel(after if step >= start else before, step)


def _has_travel_node(node):
    """True when a parsed travel tree holds at least one group: a (before, after,
    step) switch, a ("blend", ...) crossfade or an ("alt", ...) alternate."""
    if isinstance(node, tuple):
        return True
    if isinstance(node, list):
        return any(_has_travel_node(n) for n in node)
    return False


def _plan_prompt_travel(text, steps_value):
    """The per-step prompts for a travel prompt text, one per global step index: a
    string for plain and switching prompts, a ("mix", [(text, weight), ...]) item
    in the crossfade window."""
    tree = _resolve_travel_step(_parse_travel_expr(text), steps_value)
    return [_render_travel(tree, i) for i in range(steps_value)]


def _has_travel_group(text):
    """True when the prompt text contains at least one travel group: a
    [before:after:step] switch, a [before:after:step:radius] crossfade or a
    [before:after] alternate."""
    if not text:
        return False
    return _has_travel_node(_parse_travel_expr(text))


def _travel_step_text(text, steps_value, step=0):
    """The prompt a travel text has at a step index, as plain text: the text itself
    when it carries no travel group, and the heavier endpoint where a crossfade
    mixes two prompts. The conditioning of a traveling prompt is built from its
    first step; the sampler swaps in the other steps from the plan."""
    if not text or not steps_value or steps_value <= 0:
        return text
    tree = _resolve_travel_step(_parse_travel_expr(text), steps_value)
    if not _has_travel_node(tree):
        return text
    item = _render_travel(tree, step)
    return item if isinstance(item, str) else max(item[1], key=lambda x: x[1])[0]
