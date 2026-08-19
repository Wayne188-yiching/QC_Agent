"""QC Agent — Spine 原始檔分析器 + 規則引擎 L1

用法:  python qc.py <path-to-skeleton.json>

判準分層:
  L1 客觀錯誤 · L2 規範違反 · L3 風格偏離 · L4 市場對照
本檔只實作 L1/L4（不需要使用者提供任何素材即可運作）。

校正紀錄（v0.2）— 以 panda.json（3D轉面，Spine 3.8.99）為回歸案例:
  * atlas-overlap  只對 region 附件成立。mesh 附件只繪製三角面，
                   多邊形打包會刻意讓包圍盒互相嵌套。
  * bone-offscreen 多組姿勢並排擺放是轉面素材的標準做法，
                   改為偵測「姿勢群集」並只報一則資訊。
  * loop-seam      必須先確認該骨骼驅動的 slot 在首尾兩端都可見，
                   否則隱藏中的部件會被誤判為爆點。

校正紀錄（v0.5）— 美術確認 panda 的接點跳躍是刻意的轉面手法:
  * loop-seam      降為 L3。真實的接點不連續不等於瑕疵：轉面骨架會刻意在
                   循環點甩動部件，觀眾讀到的是「角色轉過去」而不是「跳格」。
                   改為附上「跳躍量 ÷ 該部件平常每格移動量」讓美術自己判斷，
                   並濾掉倍數 < 1.5 的（比它自己的運動還小，看不出來）。
  * qc-baseline    新增基準檔機制。--accept <rule> 把已確認為刻意的判定寫進
                   資產目錄的 qc-baseline.json，之後降為「參考」並保留原始
                   判定文字，讓後面接手的人知道當初量到什麼。
"""
import io
import json
import math
import os
import sys
from collections import defaultdict

SEV = {1: "必修", 2: "要修", 3: "確認", 4: "參考"}
findings = []


def flag(sev, rule, target, msg, t=None, anim=None, bones=None):
    findings.append(dict(sev=sev, rule=rule, target=target, msg=msg, t=t,
                         anim=anim, bones=bones or []))


# ── atlas ────────────────────────────────────────────────────────────
PAGE_KEYS = {"size", "format", "filter", "repeat", "pma", "scale"}
REGION_KEYS = {"index", "bounds", "offsets", "rotate",
               "xy", "size", "orig", "offset", "split", "pad"}


def parse_atlas(path):
    """Parse both atlas dialects into one shape.

    legacy (Spine <=3.8):  xy / size / orig / offset, regions indented
    modern (Spine 4.x):    bounds / offsets, regions NOT indented
    Returns (page_w, page_h, [region], page_props).
    """
    regions, cur, page = [], None, {}
    with io.open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    seen_image = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        key = s.split(":", 1)[0].strip() if ":" in s else None
        if not seen_image and key not in PAGE_KEYS and key not in REGION_KEYS:
            seen_image = True           # the page image filename
            continue
        if key in PAGE_KEYS and cur is None:
            page[key] = s.split(":", 1)[1].strip()
        elif key in REGION_KEYS and cur is not None:
            cur[key] = s.split(":", 1)[1].strip()
        else:
            cur = {"name": s}
            regions.append(cur)

    pw, ph = (int(n) for n in page.get("size", "0,0").split(","))
    for r in regions:
        if "bounds" in r:               # modern
            x, y, w, h = (int(n) for n in r["bounds"].split(","))
            ox, oy, ow, oh = (int(n) for n in r.get(
                "offsets", "0,0,{},{}".format(w, h)).split(","))
            r["xy"], r["size"], r["orig"], r["offset"] = [x, y], [w, h], [ow, oh], [ox, oy]
        else:                           # legacy
            for k in ("xy", "size", "orig", "offset"):
                r[k] = [int(n) for n in r.get(k, "0,0").split(",")]
        rot = r.get("rotate", "false")
        r["rot"] = rot
        w, h = r["size"]
        # rotate:true == 90deg. 90/270 swap the packed footprint; 180 does not.
        r["packed"] = (h, w) if rot in ("true", "90", "270") else (w, h)
    return pw, ph, regions, page


def load_alpha(png_path):
    """Alpha channel of the atlas page, or None when Pillow is unavailable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not os.path.exists(png_path):
        return None
    return Image.open(png_path).convert("RGBA")


def opaque_fraction(img, box, thr=8):
    box = (max(0, box[0]), max(0, box[1]),
           min(img.size[0], box[2]), min(img.size[1], box[3]))
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0.0
    crop = img.getchannel("A").crop(box)
    return sum(crop.histogram()[thr:]) / (crop.size[0] * crop.size[1])


def check_atlas(pw, ph, regions, region_backed, page=None, page_name="atlas"):
    """region_backed: set of atlas region names used by non-mesh attachments.
    page: PIL RGBA image of the atlas, or None (pixel checks are then skipped)."""
    used = sum(r["size"][0] * r["size"][1] for r in regions)
    total = pw * ph
    eff = used / total * 100

    rects = {r["name"]: (r["xy"][0], r["xy"][1],
                         r["xy"][0] + r["packed"][0],
                         r["xy"][1] + r["packed"][1]) for r in regions}
    names = list(rects)
    overlaps = []
    for i in range(len(names)):
        ax0, ay0, ax1, ay1 = rects[names[i]]
        for jj in range(i + 1, len(names)):
            bx0, by0, bx1, by1 = rects[names[jj]]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                overlaps.append((names[i], names[jj]))

    polygon_packed = len(overlaps) > len(regions) * 0.25

    print("\n### 圖集")
    print("  頁面 {}x{} · {} 個區塊 · 包圍盒佔用 {:.1f}%".format(
        pw, ph, len(regions), eff))
    print("  包圍盒重疊 {} 對 → 打包方式判定：{}".format(
        len(overlaps), "多邊形打包（mesh 嵌套）" if polygon_packed else "矩形打包"))

    if polygon_packed:
        flag(4, "atlas-polygon-packed", page_name,
             "圖集為多邊形打包，{} 對包圍盒互相嵌套。這是 mesh 附件的正常做法，"
             "但代表任何以 region 附件引用的區塊都會吃到鄰居的像素".format(len(overlaps)))

    # A region attachment draws its whole rect. Overlap only matters when the
    # shared zone actually holds opaque pixels — a transparent corner is fine.
    for a, b in overlaps:
        if a not in region_backed and b not in region_backed:
            continue
        ax0, ay0, ax1, ay1 = rects[a]
        bx0, by0, bx1, by1 = rects[b]
        zone = (max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1))
        zw, zh = zone[2] - zone[0], zone[3] - zone[1]

        if page is None:
            flag(3, "atlas-overlap-region", a + " / " + b,
                 "region 附件的包圍盒與鄰居重疊 {}x{}，但沒有 Pillow 無法確認"
                 "重疊區是否有內容".format(zw, zh))
            continue

        frac = opaque_fraction(page, zone)
        if frac < 0.02:
            continue  # transparent corner — harmless
        both = a in region_backed and b in region_backed
        who = a if a in region_backed else b
        flag(1 if both else 2, "atlas-overlap-region", a + " / " + b,
             "{} 以 region 附件繪製（會畫出整個矩形），重疊區 {}x{} 有 {:.0f}% 不透明像素，"
             "畫面上會看到對方的內容".format(who, zw, zh, frac * 100))

    if not polygon_packed and eff < 65:
        flag(3, "atlas-efficiency", page_name,
             "圖集佔用率僅 {:.1f}%，浪費 {:.2f} MB 顯存（RGBA8888 未壓縮）"
             .format(eff, (total - used) * 4 / 1024 / 1024))

    for r in regions:
        x, y = r["xy"]
        w, h = r["packed"]
        if x < 1 or y < 1 or x + w > pw - 1 or y + h > ph - 1:
            flag(2, "atlas-edge", r["name"],
                 "貼齊圖集邊界，Linear 過濾會取樣到頁面外，邊緣可能出現破線")

    # Same dimensions is only a hint. Confirm against pixels before calling it
    # a duplicate — 1_eyes / 1_1eyes are the same size but different artwork.
    bysize = defaultdict(list)
    for r in regions:
        bysize[tuple(r["orig"])].append(r["name"])
    for orig, nm in bysize.items():
        if len(nm) < 2:
            continue
        if page is None:
            flag(4, "atlas-same-size", " / ".join(nm),
                 "{} 個區塊尺寸相同（{}x{}），沒有 Pillow 無法確認是否為重複資產"
                 .format(len(nm), orig[0], orig[1]))
            continue
        from PIL import ImageChops
        base = page.crop(rects[nm[0]])
        for other in nm[1:]:
            cand = page.crop(rects[other])
            if cand.size != base.size:
                continue
            for label, img in (("", base), ("（旋轉 180°）", base.rotate(180))):
                if ImageChops.difference(img, cand).getbbox() is None:
                    flag(3, "atlas-duplicate", nm[0] + " / " + other,
                         "兩個區塊像素完全相同{}，可省 {:,} px²"
                         .format(label, orig[0] * orig[1]))
                    break


# ── skeleton ─────────────────────────────────────────────────────────
def world_positions(bones):
    by_name = {b["name"]: b for b in bones}
    cache = {}

    def solve(name):
        if name in cache:
            return cache[name]
        b = by_name[name]
        lx, ly = b.get("x", 0.0), b.get("y", 0.0)
        rot = b.get("rotation", 0.0)
        sx, sy = b.get("scaleX", 1.0), b.get("scaleY", 1.0)
        p = b.get("parent")
        if p is None:
            res = (lx, ly, rot, sx, sy)
        else:
            px, py, prot, psx, psy = solve(p)
            a = math.radians(prot)
            cos, sin = math.cos(a), math.sin(a)
            res = (px + lx * psx * cos - ly * psy * sin,
                   py + lx * psx * sin + ly * psy * cos,
                   prot + rot, psx * sx, psy * sy)
        cache[name] = res
        return res

    return {b["name"]: solve(b["name"]) for b in bones}


# ponytail: 固定 500 單位的分組間距，對「多姿勢並排」的素材夠用。
# 若遇到單體很寬的素材誤分組，改成依素材寬度比例計算。
def cluster_1d(values, gap=500.0):
    """Group sorted values into clusters separated by more than `gap`."""
    if not values:
        return []
    vs = sorted(values)
    out, cur = [], [vs[0]]
    for v in vs[1:]:
        if v - cur[-1] > gap:
            out.append(cur)
            cur = [v]
        else:
            cur.append(v)
    out.append(cur)
    return out


def check_skeleton(sk, bones, slots, skins):
    print("\n### 骨架")
    w = sk.get("width", 0)
    if w:
        print("  匯出邊界  x={:.0f} y={:.0f} w={:.0f} h={:.0f}".format(
            sk.get("x", 0), sk.get("y", 0), w, sk.get("height", 0)))
    else:
        print("  匯出邊界  未寫入（沒有動畫時 Spine 不計算邊界）")
    print("  {} bones · {} slots · Spine {}".format(
        len(bones), len(slots), sk.get("spine")))

    wp = world_positions(bones)
    clusters = cluster_1d([p[0] for p in wp.values()])
    if len(bones) > 1:
        print("  姿勢群集 {} 組: {}".format(
            len(clusters),
            " / ".join("x[{:.0f}..{:.0f}] n={}".format(c[0], c[-1], len(c))
                       for c in clusters)))

    if len(clusters) > 1:
        widest = max(c[-1] - c[0] for c in clusters)
        flag(4, "skeleton-pose-clusters", "skeleton",
             "骨骼分成 {} 組並排擺放（轉面素材常態），單組實際只佔 {:.0f} 寬，"
             "但匯出邊界被撐到 {:.0f}。若 runtime 用邊界做剔除或自動置中，記得改用單組範圍"
             .format(len(clusters), widest, w))

    attach_map = defaultdict(set)
    for skin in skins:
        for slot_name, atts in skin.get("attachments", {}).items():
            attach_map[slot_name].update(atts.keys())

    empty = [s["name"] for s in slots if not attach_map.get(s["name"])]
    if empty:
        flag(3, "slot-empty", ", ".join(empty),
             "{} 個 slot 在 skin 中沒有任何 attachment，是殘留還是刻意留空？".format(len(empty)))

    parents = {b.get("parent") for b in bones}
    slot_bones = {s["bone"] for s in slots}
    dead = [b["name"] for b in bones
            if b["name"] not in parents and b["name"] not in slot_bones]
    if dead:
        flag(3, "bone-unused", ", ".join(dead),
             "{} 根骨骼既沒有子骨骼也沒有掛 slot，是空骨".format(len(dead)))
    return attach_map


# ── animations ───────────────────────────────────────────────────────
CURVE_LINEAR, CURVE_STEPPED, CURVE_BEZIER = "linear", "stepped", "bezier"
VALUE_FIELDS = {"rotate": ("angle",), "translate": ("x", "y"),
                "scale": ("x", "y"), "shear": ("x", "y")}
LOOPING = ("idle", "idle_slow")


def curve_of(key):
    c = key.get("curve")
    if c is None:
        return CURVE_LINEAR
    return CURVE_STEPPED if c == "stepped" else CURVE_BEZIER


def anim_duration(anim):
    dur = 0.0
    def scan(keys):
        nonlocal dur
        for k in keys:
            dur = max(dur, k.get("time", 0.0))
    for tls in anim.get("bones", {}).values():
        for keys in tls.values():
            scan(keys)
    for tls in anim.get("slots", {}).values():
        for keys in tls.values():
            scan(keys)
    for skin in anim.get("deform", {}).values():
        for slot in skin.values():
            for keys in slot.values():
                scan(keys)
    scan(anim.get("drawOrder", []))
    scan(anim.get("events", []))
    return dur


def bone_slot_index(bones, slots):
    """bone -> set of slots driven by it or any descendant bone."""
    children = defaultdict(list)
    for b in bones:
        if b.get("parent"):
            children[b["parent"]].append(b["name"])
    direct = defaultdict(set)
    for s in slots:
        direct[s["bone"]].add(s["name"])

    memo = {}
    def collect(name):
        if name in memo:
            return memo[name]
        memo[name] = set()          # guard against cycles
        out = set(direct.get(name, ()))
        for c in children.get(name, ()):
            out |= collect(c)
        memo[name] = out
        return out

    return {b["name"]: collect(b["name"]) for b in bones}


def attachment_at(anim, slot, setup_attachment, t):
    """Resolve which attachment a slot shows at time t (None == hidden)."""
    keys = anim.get("slots", {}).get(slot, {}).get("attachment")
    if not keys:
        return setup_attachment
    cur = setup_attachment
    for k in keys:
        if k.get("time", 0.0) <= t + 1e-6:
            cur = k.get("name")
        else:
            break
    return cur


def alpha_at(anim, slot, setup_color, t):
    """Slot alpha at time t, 0..1. A slot faded to 0 cannot pop visibly."""
    def a_of(hex8):
        return int(hex8[6:8], 16) / 255.0 if hex8 and len(hex8) >= 8 else 1.0

    keys = anim.get("slots", {}).get(slot, {}).get("color")
    if not keys:
        return a_of(setup_color)
    cur = a_of(setup_color)
    for k in keys:
        if k.get("time", 0.0) <= t + 1e-6:
            cur = a_of(k.get("color"))
        else:
            break
    return cur


def slot_visible(anim, slot, setup_attachment, setup_color, t):
    return (attachment_at(anim, slot, setup_attachment, t) is not None
            and alpha_at(anim, slot, setup_color, t) > 0.02)


def check_animations(anims, bones, slots, attach_map, baked):
    b2s = bone_slot_index(bones, slots)
    setup = {s["name"]: s.get("attachment") for s in slots}
    setup_col = {s["name"]: s.get("color", "ffffffff") for s in slots}

    print("\n### 動畫")
    print("  {:<11} {:>6} {:>6} {:>8} {:>8} {:>7} {:>7}".format(
        "name", "長度", "keys", "linear", "bezier", "stepped", "deform"))
    for name, anim in anims.items():
        dur = anim_duration(anim)
        counts = defaultdict(int)
        total_keys = 0
        seams = []

        for bone, tls in anim.get("bones", {}).items():
            # A bone may itself be hidden while still carrying a visible child —
            # name that child, so the finding says what the viewer actually sees.
            shown = [s for s in sorted(b2s.get(bone, ()))
                     if slot_visible(anim, s, setup.get(s), setup_col.get(s), 0.0)
                     and slot_visible(anim, s, setup.get(s), setup_col.get(s), dur)]
            visible_ends = bool(shown)
            carries = shown[0] if shown else None

            for prop, keys in tls.items():
                for k in keys:
                    counts[curve_of(k)] += 1
                    total_keys += 1
                if not visible_ends or prop not in VALUE_FIELDS or len(keys) < 2:
                    continue
                first, last = keys[0], keys[-1]
                for f in VALUE_FIELDS[prop]:
                    d = last.get(f, 0.0) - first.get(f, 0.0)
                    if abs(d) > 0.5:
                        seams.append((bone, prop, f, first.get(f, 0.0),
                                      last.get(f, 0.0), d, carries))

        deform_keys = sum(len(keys)
                          for skin in anim.get("deform", {}).values()
                          for slot in skin.values()
                          for keys in slot.values())
        lin = counts[CURVE_LINEAR]
        pct = (lin / total_keys * 100) if total_keys else 0.0

        print("  {:<11} {:>5.2f}s {:>6} {:>7.0f}% {:>8} {:>7} {:>7}".format(
            name, dur, total_keys, pct,
            counts[CURVE_BEZIER], counts[CURVE_STEPPED], deform_keys))

        if total_keys and pct > 80:
            flag(3, "curve-linear-ratio", name,
                 "{} 的 {} 個關鍵影格中 {:.0f}% 是線性插值。轉面類動畫線性比例天生偏高，"
                 "但仍值得確認可見段落有沒有緩動".format(name, total_keys, pct))

        if dur == 0 and total_keys:
            flag(1, "anim-zero-length", name, "動畫長度為 0")

        if name in LOOPING:
            check_loop_seam(name, baked.get(name), dur)


# ── pipeline ─────────────────────────────────────────────────────────
def load_skeleton(path):
    """Load a skeleton, flagging a binary-named file that is really JSON."""
    with io.open(path, "rb") as f:
        head = f.read(4)
    is_json = head[:1] in (b"{", b"\xef")
    claims_binary = path.lower().endswith((".skel", ".skel.bytes", ".bytes"))

    if claims_binary and is_json:
        flag(1, "skel-format-mismatch", os.path.basename(path),
             "副檔名宣告為 Spine 二進位，內容卻是 JSON（前 4 byte = {!r}）。"
             "Unity 用 SkeletonBinary 載入會直接拋例外，必須改用 SkeletonJson "
             "或重新以 Binary 格式匯出".format(head))
    if not is_json:
        raise SystemExit("這是 Spine 二進位檔，本版尚未支援解碼。請匯出 JSON 後再跑。")

    with io.open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def check_pipeline(sk_path, j, page_props):
    """Cross-file consistency: stale exports, absolute paths, packer settings."""
    sk = j.get("skeleton", {})
    print("\n### 流程")
    print("  Spine {} · 圖集 pma={} scale={}".format(
        sk.get("spine"), page_props.get("pma", "false"), page_props.get("scale", "1")))

    imgs = sk.get("images", "")
    if imgs and (":" in imgs[:3] or imgs.startswith("\\\\")):
        drive = imgs[:2]
        here = os.path.abspath(sk_path)[:2]
        if drive.lower() != here.lower():
            flag(3, "image-path-absolute", os.path.basename(sk_path),
                 "來源圖層路徑是絕對路徑且指向 {} 槽（{}），但檔案在 {} 槽。"
                 "換機器或換人接手時無法重新匯入".format(drive, imgs, here))

    try:
        scale = float(page_props.get("scale", "1"))
    except ValueError:
        scale = 1.0
    if abs(scale - 1.0) > 1e-6:
        flag(3, "atlas-scale", "atlas",
             "圖集以 {:.0%} 縮放匯出。附件尺寸是原始值、貼圖是縮小版，"
             "放大到原尺寸顯示會軟掉；若這是刻意的省記憶體設定就忽略".format(scale))

    # stale export: is the .spine project newer than what was exported?
    exp_dir = os.path.dirname(os.path.abspath(sk_path))
    newest_src, newest_t = None, 0.0
    for root in (exp_dir, os.path.dirname(exp_dir)):
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            p = os.path.join(root, entry)
            if entry.lower().endswith((".spine", ".psd")) and os.path.isfile(p):
                t = os.path.getmtime(p)
                if t > newest_t:
                    newest_src, newest_t = p, t
    if newest_src:
        exp_t = os.path.getmtime(sk_path)
        days = (newest_t - exp_t) / 86400.0
        if days > 0.5:
            import datetime
            fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
            flag(1, "stale-export", os.path.basename(sk_path),
                 "來源 {} 修改於 {}，但匯出檔停在 {}，落後 {:.0f} 天。"
                 "現在測的很可能不是最新版".format(
                     os.path.basename(newest_src), fmt(newest_t), fmt(exp_t), days))


def check_rigging(j):
    bones, slots = j["bones"], j["slots"]
    anims = j.get("animations", {})

    live = {n: a for n, a in anims.items()
            if a.get("bones") or a.get("slots") or a.get("deform")
            or a.get("drawOrder") or a.get("events")}
    if not live:
        flag(1, "no-animation", "skeleton",
             "{} 個動畫全部是空的（{}），這份匯出沒有任何關鍵影格".format(
                 len(anims), ", ".join(anims) or "一個都沒有") if anims else
             "沒有任何動畫")

    if len(bones) <= 1 and len(slots) > 3:
        flag(2, "unrigged", "skeleton",
             "{} 個 slot 全部掛在 root，只有 1 根骨頭。這是分層匯入後尚未綁定的狀態，"
             "不是可動的骨架".format(len(slots)))
    return live


# ── main ─────────────────────────────────────────────────────────────
def find_atlas(sk_path):
    base = os.path.splitext(sk_path)[0]
    for cand in (base + ".atlas", base + ".atlas.txt",
                 os.path.splitext(base)[0] + ".atlas",
                 os.path.splitext(base)[0] + ".atlas.txt"):
        if os.path.exists(cand):
            return cand
    d = os.path.dirname(os.path.abspath(sk_path))
    for entry in sorted(os.listdir(d)):
        if entry.endswith((".atlas", ".atlas.txt")):
            return os.path.join(d, entry)
    return None


def main(json_path, html_out=None, accept=None):
    j = load_skeleton(json_path)
    atlas_path = find_atlas(json_path)
    atlas_desc = "(無)"

    print("=" * 74)
    print("QC Agent · L1 規則包 v0.4")
    print("資產 " + os.path.basename(json_path))
    print("=" * 74)

    # which atlas regions are drawn as whole rects (region attachments)?
    region_backed, kinds = set(), defaultdict(int)
    for skin in j["skins"]:
        for slot, atts in skin.get("attachments", {}).items():
            for an, a in atts.items():
                kind = a.get("type", "region")
                kinds[kind] += 1
                if kind == "region":
                    region_backed.add(a.get("path", an))
    print("\n附件型別: " + " · ".join("{} {}".format(k, v) for k, v in kinds.items()))

    page_props = {}
    if atlas_path:
        pw, ph, regions, page_props = parse_atlas(atlas_path)
        png_path = os.path.join(os.path.dirname(atlas_path),
                                os.path.splitext(os.path.basename(atlas_path))[0])
        png_path = os.path.splitext(png_path)[0] + ".png"
        page = load_alpha(png_path)
        if page is None:
            print("(找不到 {} 或未安裝 Pillow，像素層檢查降級為推測)".format(
                os.path.basename(png_path)))
        check_atlas(pw, ph, regions, region_backed, page,
                    os.path.basename(png_path))
        atlas_desc = "{}x{} · {} 區塊".format(pw, ph, len(regions))
    else:
        print("\n(找不到 .atlas，跳過圖集檢查)")

    check_pipeline(json_path, j, page_props)
    check_rigging(j)
    attach_map = check_skeleton(j["skeleton"], j["bones"], j["slots"], j["skins"])
    baked = {}
    for _name in j.get("animations", {}):
        try:
            baked[_name] = bake(j, _name)
        except Exception as _exc:
            print("  (無法烘焙 {}：{})".format(_name, _exc))
    check_animations(j["animations"], j["bones"], j["slots"], attach_map, baked)

    if accept:
        write_baseline(json_path, set(accept))
    apply_baseline(json_path)

    print("\n" + "=" * 74)
    print("FINDINGS")
    print("=" * 74)
    for n, f in enumerate(sorted(findings, key=lambda f: f["sev"]), 1):
        head = "[{}] {} · {}".format(SEV[f["sev"]], f["rule"], f["target"])
        if f["t"] is not None:
            head += "  @ {:.2f}s".format(f["t"])
        print("\n{:>2}. {}".format(n, head))
        print("    " + f["msg"])

    tally = defaultdict(int)
    for f in findings:
        tally[f["sev"]] += 1
    print("\n" + "-" * 74)
    print("合計  " + (" · ".join("{} {}".format(SEV[s], tally[s])
                                 for s in sorted(SEV) if tally[s]) or "無"))

    if html_out:
        emit_html(html_out, json_path, j, atlas_desc, baked)




# ── baking: skeleton -> intermediate format ──────────────────────────
# This is the 中介格式 the viewer consumes. Any engine that can emit
#   bones:  [{name, parent, frames:[[x, y, angleDeg, length], ...]}]
#   slots:  {name: [0|1 per frame]}
# can drive the same viewer. Spine is just the first producer.

def _bezier_lut(cx1, cy1, cx2, cy2, n=17):
    """Sample a CSS-style cubic bezier into an (x, y) lookup table."""
    lut = []
    for i in range(n):
        s = i / (n - 1)
        u = 1 - s
        x = 3 * u * u * s * cx1 + 3 * u * s * s * cx2 + s ** 3
        y = 3 * u * u * s * cy1 + 3 * u * s * s * cy2 + s ** 3
        lut.append((x, y))
    return lut


def _ease(key, frac):
    """Map linear 0..1 progress through this key's outgoing curve."""
    c = key.get("curve")
    if c is None:
        return frac
    if c == "stepped":
        return 0.0
    if isinstance(c, list) and len(c) >= 4:
        lut = _bezier_lut(*c[:4])
    else:                                   # 3.8 writes curve/c2/c3/c4
        lut = _bezier_lut(c, key.get("c2", 0.0),
                          key.get("c3", 1.0), key.get("c4", 1.0))
    for i in range(1, len(lut)):
        if lut[i][0] >= frac:
            x0, y0 = lut[i - 1]
            x1, y1 = lut[i]
            span = x1 - x0
            return y0 if span <= 1e-9 else y0 + (y1 - y0) * (frac - x0) / span
    return 1.0


def sample_timeline(keys, fields, defaults, t):
    """Value of a bone timeline at time t, as a tuple matching `fields`."""
    if not keys:
        return defaults
    if t <= keys[0].get("time", 0.0):
        return tuple(keys[0].get(f, d) for f, d in zip(fields, defaults))
    if t >= keys[-1].get("time", 0.0):
        return tuple(keys[-1].get(f, d) for f, d in zip(fields, defaults))
    for i in range(len(keys) - 1):
        t0 = keys[i].get("time", 0.0)
        t1 = keys[i + 1].get("time", 0.0)
        if t0 <= t <= t1:
            span = t1 - t0
            frac = 0.0 if span <= 1e-9 else (t - t0) / span
            e = _ease(keys[i], frac)
            out = []
            for f, d in zip(fields, defaults):
                v0 = keys[i].get(f, d)
                v1 = keys[i + 1].get(f, d)
                out.append(v0 + (v1 - v0) * e)
            return tuple(out)
    return defaults


def bake(j, anim_name, fps=30):
    """Bake one animation into per-frame bone transforms + slot visibility."""
    bones = j["bones"]
    slots = j["slots"]
    anim = j["animations"][anim_name]
    dur = anim_duration(anim)
    n = max(2, int(round(dur * fps)) + 1)
    btl = anim.get("bones", {})

    setup = {s["name"]: s.get("attachment") for s in slots}
    setup_col = {s["name"]: s.get("color", "ffffffff") for s in slots}

    order = [b["name"] for b in bones]
    by_name = {b["name"]: b for b in bones}
    frames = {name: [] for name in order}

    for i in range(n):
        t = dur * i / (n - 1)
        world = {}
        for name in order:                       # parents precede children
            b = by_name[name]
            tl = btl.get(name, {})
            dx, dy = sample_timeline(tl.get("translate"), ("x", "y"), (0.0, 0.0), t)
            (da,) = sample_timeline(tl.get("rotate"), ("angle",), (0.0,), t)
            sx, sy = sample_timeline(tl.get("scale"), ("x", "y"), (1.0, 1.0), t)

            lx = b.get("x", 0.0) + dx
            ly = b.get("y", 0.0) + dy
            lr = b.get("rotation", 0.0) + da
            lsx = b.get("scaleX", 1.0) * sx
            lsy = b.get("scaleY", 1.0) * sy

            p = b.get("parent")
            if p is None or p not in world:
                wx, wy, wr, wsx, wsy = lx, ly, lr, lsx, lsy
            else:
                px, py, pr, psx, psy = world[p]
                a = math.radians(pr)
                cos, sin = math.cos(a), math.sin(a)
                wx = px + (lx * psx) * cos - (ly * psy) * sin
                wy = py + (lx * psx) * sin + (ly * psy) * cos
                wr, wsx, wsy = pr + lr, psx * lsx, psy * lsy
            world[name] = (wx, wy, wr, wsx, wsy)
            frames[name].append([round(wx, 1), round(wy, 1), round(wr, 1),
                                 round(b.get("length", 0.0) * abs(wsx), 1)])

    vis = {}
    for s in slots:
        nm = s["name"]
        row = []
        for i in range(n):
            t = dur * i / (n - 1)
            row.append(1 if slot_visible(anim, nm, setup.get(nm),
                                         setup_col.get(nm), t) else 0)
        if any(row):
            vis[nm] = row

    # Reference scale: how tall the visible figure actually is on screen.
    # Raw Spine units mean nothing to an artist — every delta gets expressed
    # against this instead ("half the character's height", not "529 units").
    heights = []
    by_bone = {b["name"]: b for b in bones}
    slot_bone = {sl["name"]: sl["bone"] for sl in slots}
    for i in range(n):
        shown = {slot_bone[sl] for sl, row in vis.items() if row[i]}
        for nm in list(shown):
            p = by_bone[nm].get("parent")
            while p and p not in shown:
                shown.add(p)
                p = by_bone[p].get("parent")
        ys = [frames[nm][i][1] for nm in shown if nm in frames]
        if len(ys) > 1:
            heights.append(max(ys) - min(ys))
    heights.sort()
    ref = heights[len(heights) // 2] if heights else 0.0

    return {
        "duration": round(dur, 3),
        "frames": n,
        "fps": fps,
        "ref": round(ref, 1),
        "bones": [{"name": nm, "parent": by_name[nm].get("parent"),
                   "f": frames[nm]} for nm in order],
        "slotBone": {s["name"]: s["bone"] for s in slots},
        "vis": vis,
    }


def emit_html(out_path, sk_path, j, atlas_desc, baked):
    """Write a self-contained report: findings + baked wireframe viewer."""
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
    with io.open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()

    kinds = defaultdict(int)
    for skin in j["skins"]:
        for atts in skin.get("attachments", {}).values():
            for a in atts.values():
                kinds[a.get("type", "region")] += 1

    payload = {
        "asset": os.path.basename(sk_path),
        "spine": j["skeleton"].get("spine", "?"),
        "bones": len(j["bones"]),
        "slots": len(j["slots"]),
        "attachments": " · ".join("{} {}".format(k, v) for k, v in kinds.items()),
        "atlas": atlas_desc,
        "animations": baked,
        "findings": sorted(findings, key=lambda f: f["sev"]),
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out = tpl.replace("/*__DATA__*/ null", blob)
    with io.open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print("\n報告已寫出: {}  ({:.0f} KB)".format(out_path, len(out) / 1024))




def live_set(b, frame):
    """Bones drawn at this frame: those carrying a visible slot, plus ancestors."""
    by = {x["name"]: x for x in b["bones"]}
    out = {b["slotBone"][sl] for sl, row in b["vis"].items() if row[frame]}
    for nm in list(out):
        p = by.get(nm, {}).get("parent")
        while p and p not in out:
            out.add(p)
            p = by.get(p, {}).get("parent")
    return out


def check_loop_seam(name, b, dur):
    """Loop discontinuity, measured on baked world positions.

    Calibration note — panda idle / idle_slow: a real seam is not automatically
    a defect. A turn rig can snap parts at the loop point on purpose and the
    viewer reads it as the character whipping around. So this reports at 確認,
    not 必修, and carries the one number that lets an artist decide fast:
    how the jump compares with that part's own per-frame motion. A jump smaller
    than the part already moves each frame is invisible and is not reported.
    """
    if not b or not b.get("ref"):
        return
    last = b["frames"] - 1
    live0, liveN = live_set(b, 0), live_set(b, last)
    ref = b["ref"]

    off = []
    for bone in b["bones"]:
        nm = bone["name"]
        if nm not in live0 or nm not in liveN:
            continue
        f = bone["f"]
        a, z = f[0], f[last]
        d = math.hypot(z[0] - a[0], z[1] - a[1])
        da = abs((z[2] - a[2] + 180) % 360 - 180)
        rel = d / ref
        if rel <= 0.02 and da <= 8:
            continue

        pre = [math.hypot(f[i + 1][0] - f[i][0], f[i + 1][1] - f[i][1])
               for i in range(max(0, last - 3), last)]
        post = [math.hypot(f[i + 1][0] - f[i][0], f[i + 1][1] - f[i][1])
                for i in range(0, min(3, last))]
        speed = max(sum(pre) / len(pre) if pre else 0.0,
                    sum(post) / len(post) if post else 0.0)
        ratio = d / speed if speed > 0.5 else float("inf")
        if ratio < 1.5:
            continue          # smaller than the part's own motion — invisible
        off.append((rel, d, da, nm, a[3], ratio))

    if not off:
        return

    off.sort(reverse=True)
    rel, d, da, nm, own, ratio = off[0]
    head = "{} 跳了角色可見高度的 {:.0f}%".format(nm, rel * 100)
    if own > 1:
        head += "（它自己長度的 {:.1f} 倍".format(d / own)
        head += "、平常每格移動量的 {:.1f} 倍）".format(ratio) if ratio != float("inf")             else "，而它在接點前後幾乎靜止）"
    if da > 8:
        head += "，同時轉了 {:.0f}°".format(da)

    rest = "；".join("{} {:.0f}%".format(o[3], o[0] * 100) for o in off[1:4])
    flag(3, "loop-seam", name,
         "循環接點首尾不接：{} 根首尾都看得見的骨骼位置對不上。{}。其次：{}。"
         "倍數越高越可能被看成瑕疵；若這是刻意的轉面手法，"
         "用 --accept loop-seam 記進基準檔就不會再報".format(
             len(off), head, rest or "無"),
         t=dur, anim=name, bones=[o[3] for o in off[:6]])


# ── baseline: findings the artist has confirmed are intentional ───────
def baseline_path(sk_path):
    return os.path.join(os.path.dirname(os.path.abspath(sk_path)), "qc-baseline.json")


def load_baseline(sk_path):
    p = baseline_path(sk_path)
    if not os.path.exists(p):
        return []
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def apply_baseline(sk_path):
    """Accepted findings drop to 參考 and carry the reason they were accepted."""
    accepted = load_baseline(sk_path)
    if not accepted:
        return
    for f in findings:
        for e in accepted:
            if e["rule"] == f["rule"] and e.get("anim", f.get("anim")) == f.get("anim"):
                f["sev"] = 4
                f["msg"] = "[已確認為刻意] {}｜原始判定：{}".format(
                    e.get("note", "無說明"), f["msg"])
                break


def write_baseline(sk_path, rules):
    p = baseline_path(sk_path)
    existing = load_baseline(sk_path)
    known = {(e["rule"], e.get("anim")) for e in existing}
    added = 0
    for f in findings:
        if f["rule"] not in rules:
            continue
        key = (f["rule"], f.get("anim"))
        if key in known:
            continue
        existing.append({"rule": f["rule"], "anim": f.get("anim"),
                         "target": f["target"], "note": "美術確認為刻意手法"})
        known.add(key)
        added += 1
    with io.open(p, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2)
    print("\n基準檔已更新: {}  (新增 {} 筆)".format(p, added))


# ponytail: 這個區塊必須留在檔尾——所有函式定義都要先於它執行。
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    html, accept = None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--html" and i + 1 < len(argv):
            html = argv[i + 1]
            if html in args:
                args.remove(html)
            i += 1
        elif argv[i] == "--accept" and i + 1 < len(argv):
            accept.append(argv[i + 1])
            if argv[i + 1] in args:
                args.remove(argv[i + 1])
            i += 1
        i += 1
    if not args:
        raise SystemExit("用法: python qc.py <skeleton> [--html out.html] "
                         "[--accept <rule>]")
    main(args[0], html, accept)
