from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
import json, re, unicodedata
from typing import List, Dict, Any

def normalize(s: str) -> str:
    val = s.get("text", "")
    val = unicodedata.normalize("NFKC", val).lower().strip()
    val = val.replace("µ", "u").replace("μ", "u")
    return val

CONNECTIVE_RE = re.compile(r"(并且|并(?!不|非)|然后|随后|接着|同时|以及|再|;| and | then )")

def is_complex(utter: str) -> bool:
    s = normalize(utter)
    if len(s) > 25: return True
    if CONNECTIVE_RE.search(s): return True
    if len(re.findall(r"(ch?\s*[1-4]|通道\s*[1-4])", s)) >= 2: return True
    if len(re.findall(r"\d+(?:\.\d+)?\s*(mv|v|kv|fs|ps|ns|us|ms|s|hz|khz|mhz|ghz)", s)) >= 2: return True
    if re.search(r"(如果|当.*?时|先.*?再)", s): return True
    return False

def score(x, y):
    return 0.6 * fuzz.token_set_ratio(x, y)/100.0 + 0.4 * fuzz.partial_ratio(x, y)/100.0

def quick_map(utter, catalog):  # catalog: list of dicts {id, canonical, aliases, regex, template}
    if is_complex(utter):
        return {"route":"LLM"}
    x = normalize(utter)

    # 候选生成（BM25）
    docs = [normalize(c["canonical"] + " " + " ".join(c.get("aliases",[]))) for c in catalog]
    bm25 = BM25Okapi([d.split() for d in docs])
    idxs = bm25.get_top_n(x.split(), list(range(len(docs))), n=10)

    # 模糊综合打分 + 正则加分
    scored = []
    for i in idxs:
        text = docs[i]
        s = score(x, text)
        for rg in catalog[i].get("regex", []):
            if re.search(rg, x):
                s += 0.08; break
        scored.append((s, i))
    scored.sort(reverse=True)

    if not scored: return {"route":"LLM"}

    best_s, best_i = scored[0]
    τ_hi, τ_lo = 0.82, 0.68
    if best_s >= τ_hi:
        item = catalog[best_i]
        return {"route":"FAST", "id": item["id"], "template": item["template"], "score": round(best_s,3)}
    elif best_s < τ_lo:
        return {"route":"LLM"}
    else:
        return {"route":"RE-RANK", "candidates":[{"id":catalog[i]["id"],"s":round(s,3)} for s,i in scored]}

def load_catalog(path: str):
    """
    返回：
      catalog: List[dict] 供 quick_map 使用
      index:   Dict[id -> full_intent] 供后续抽槽/渲染用（包含 slots/constraints 等）
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intents = data["intents"]

    catalog = []
    index = {}
    for it in intents:
        rid = it["id"]
        canonical = it["canonical"]
        aliases = it.get("aliases", [])
        # 编译正则，大小写不敏感
        regex_list = [re.compile(p, re.I) for p in it.get("regex", [])]
        template = it["scpi"]["template"] if "scpi" in it and "template" in it["scpi"] else it.get("template", "")

        catalog.append({
            "id": rid,
            "canonical": canonical,
            "aliases": aliases,
            "regex": regex_list,
            "template": template
        })
        index[rid] = it  # 保存完整定义，后面抽槽/校验/确认要用
    return catalog, index

# ---------- 槽位抽取 & 模板渲染 ----------
def infer_on_off(utter_norm: str):
    # 简单根据词判断状态
    if re.search(r"(打开|开启|enable|on)", utter_norm): return "ON"
    if re.search(r"(关闭|关掉|disable|off)", utter_norm): return "OFF"
    return None

def extract_slots(utter: str, intent_def: Dict[str, Any]):
    """
    1) 尝试匹配任一 regex：
       - 若有命名组：直接取 groupdict()
       - 若 slots.channel 有 from_group：按索引取
    2) 补默认值/归一化（如 µs/μs→us，unit 默认）
    3) 对 ON/OFF 类从话语推断
    """
    s = normalize(utter)
    slots = {}
    matched = None
    m = None
    patterns: List[re.Pattern] = [re.compile(p, re.I) if isinstance(p, str) else p for p in intent_def.get("regex", [])]
    for rg in patterns:
        m = rg.search(s)
        if m:
            matched = rg
            break

    slot_spec = intent_def.get("slots", {}) or {}

    if m:
        # 先取命名组
        gd = {k: v for k, v in m.groupdict().items() if v is not None}
        slots.update(gd)

        # 再用 from_group 抽未命名组（例如 channel）
        for k, spec in slot_spec.items():
            if k in slots:
                continue
            gidx = spec.get("from_group")
            if gidx:
                for idx in gidx:  # 允许多个候选组编号，取第一个有值的
                    try:
                        val = m.group(idx)
                    except IndexError:
                        val = None
                    if val:
                        slots[k] = val
                        break

    # ON/OFF 推断
    if "state" in slot_spec and "state" not in slots:
        st = infer_on_off(s)
        if st: slots["state"] = st

    # 归一化 & 默认值
    for k, spec in slot_spec.items():
        if k not in slots:
            if "default" in spec:
                slots[k] = spec["default"]
            continue
        # normalize map
        norm_map = spec.get("normalize")
        if norm_map and isinstance(slots[k], str):
            slots[k] = norm_map.get(slots[k], slots[k])
        # 枚举大小写统一
        if spec.get("type") == "enum" and isinstance(slots[k], str):
            slots[k] = slots[k].lower()

    # 简单范围校验（可按需提升为严格异常）
    constraints = intent_def.get("constraints", {}) or {}
    rng = constraints.get("range", {})
    for k, rr in rng.items():
        if k in slots:
            try:
                val = float(slots[k])
                if "min" in rr and val < rr["min"]:
                    raise ValueError(f"{k} below min")
                if "max" in rr and val > rr["max"]:
                    raise ValueError(f"{k} above max")
            except Exception:
                # 不通过就返回缺槽或让上游澄清
                return None

    return slots

def render_scpi(template: str, slots: Dict[str, Any]):
    # 简单占位渲染：{value}{unit} 等
    try:
        return template.format(**slots)
    except KeyError:
        # 有占位但没槽位，交由上游澄清/LLM
        return None

# 封装：先 quick_map，再抽槽&渲染 ----------
def map_and_render(utter: str):
    catalog, index = load_catalog("fast_command.json")
    print(utter)

    route = quick_map(utter, catalog)
    if route.get("route") != "FAST":
        return route

    intent_id = route["id"]
    intent_def = index[intent_id]
    slots = extract_slots(utter, intent_def)
    if slots is None and "{" in route["template"]:
        # 模板有占位但没抽出槽位 → 让上游澄清/回退
        return {"route":"LLM", "reason":"missing_slots", "id": intent_id}

    scpi = render_scpi(route["template"], slots or {})
    if scpi is None and "{" in route["template"]:
        return {"route":"LLM", "reason":"render_failed", "id": intent_id}

    return {
        "route": "FAST",
        "id": intent_id,
        "score": route["score"],
        "slots": slots or {},
        "scpi": scpi if scpi else route["template"]
    }
