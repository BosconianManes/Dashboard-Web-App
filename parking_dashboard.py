import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SmartPark IoT", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# 1. STYLE (matches the reference: light bg, white cards, blue accent)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #F8FAFC; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem !important; padding-left: 2rem; padding-right: 2rem; max-width: 1400px; }

/* sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; padding: 0.55rem 0.75rem; }
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] { background: #2563EB; border: none; color: #fff; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; margin-top: 8px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { width: 100%; padding: 10px 12px; border-radius: 8px; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #F1F5F9; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: #EFF6FF; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #2563EB; font-weight: 600; }
section[data-testid="stSidebar"] div[role="radiogroup"] p { color: #475569; font-size: 14px; }

/* top bar */
.topbar { display: flex; justify-content: space-between; align-items: center; background: #fff; border: 1px solid #E2E8F0;
          border-radius: 12px; padding: 14px 22px; margin-bottom: 20px; }
.brand { font-size: 20px; font-weight: 700; color: #0F172A; letter-spacing: .5px; }
.brand span { color: #8B1A1A; }
.brand em { font-style: normal; font-weight: 500; font-size: 15px; color: #64748B; margin-left: 10px; }
.topmeta { color: #64748B; font-size: 13px; }

/* page header */
.page-title { font-size: 30px; font-weight: 700; color: #0F172A; margin: 0 0 6px; line-height: 1.2; }
.status { display: flex; gap: 12px; align-items: center; background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 10px 16px; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 0 4px #DCFCE7; flex: none; }
.s-title { font-weight: 600; color: #0F172A; font-size: 14px; }
.s-sub { color: #64748B; font-size: 12px; }

/* stat cards */
.stat { display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; }
.num { font-size: 22px; font-weight: 700; color: #0F172A; line-height: 1.15; }
.lbl { font-size: 13px; color: #64748B; }

/* cards (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid #E2E8F0 !important; border-radius: 12px;
          box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.card-title { font-size: 17px; font-weight: 700; color: #0F172A; margin: 8px 0 6px; }
.big { font-size: 15px; color: #64748B; margin: 4px 0 2px; }
.big b { font-size: 24px; color: #2563EB; margin: 0 18px 0 6px; font-weight: 600; }
.big b.gray { color: #475569; }

/* list rows */
.row { display: flex; align-items: center; gap: 14px; padding: 10px 0; }
.av { width: 46px; height: 46px; border-radius: 10px; background: #EFF6FF; display: flex; align-items: center; justify-content: center; font-size: 24px; flex: none; }
.mid { flex: 1; min-width: 0; }
.name { font-weight: 600; color: #0F172A; font-size: 14px; }
.sub { font-size: 12px; color: #94A3B8; }
.badge { background: #F1F5F9; color: #475569; border-radius: 999px; padding: 3px 12px; font-size: 12px; white-space: nowrap; }
.right { text-align: right; font-size: 12px; }
.right .amt { color: #0F172A; font-weight: 600; font-size: 13px; }
.s-parked { color: #2563EB; } .s-over { color: #DC2626; } .s-exit { color: #64748B; }
.empty { color: #94A3B8; font-size: 13px; padding: 24px 0; text-align: center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
# 2. CONFIG & MOCK DATA
# =====================================================================
VEHICLE_RATES = {"Motorcycle": 20, "Sedan": 50, "Hatchback": 45, "SUV": 80, "Pickup": 90, "Van": 100, "Truck": 150}
ICONS = {"Motorcycle": "🏍️", "Sedan": "🚗", "Hatchback": "🚘", "SUV": "🚙", "Pickup": "🛻", "Van": "🚐", "Truck": "🚚"}
MODELS = {
    "Motorcycle": ["Honda Click 125", "Yamaha Mio i125", "Suzuki Raider"],
    "Sedan": ["Toyota Vios", "Honda City", "Mitsubishi Mirage G4"],
    "Hatchback": ["Toyota Wigo", "Suzuki Celerio", "Honda Jazz"],
    "SUV": ["Toyota Fortuner", "Ford Everest", "Mitsubishi Montero Sport"],
    "Pickup": ["Toyota Hilux", "Nissan Navara", "Isuzu D-Max"],
    "Van": ["Toyota Hiace", "Nissan Urvan", "Hyundai Starex"],
    "Truck": ["Isuzu Elf", "Mitsubishi Fuso Canter", "Hino 300"],
}
PLATES = ["ABC-1234", "XYZ-9876", "NCR-555", "QWE-1122", "JKL-3344", "MNO-7788", "PQR-9900"]

# Assumed overnight violation fee (PHP). Adjust to the establishment's policy.
VIOLATION_FEE = 500


def make_vehicle(plate, vtype, time_in):
    return {
        "plate": plate, "type": vtype, "model": random.choice(MODELS[vtype]),
        "rate": VEHICLE_RATES[vtype], "time_in": time_in,
        "confidence": round(random.uniform(0.88, 0.99), 3),  # simulated classifier confidence
        "response_ms": random.randint(180, 450),             # simulated barrier response time
    }


def generate_mock_vehicle(overnight=False):
    vtype = random.choice(list(VEHICLE_RATES))
    parked = {v["plate"] for v in st.session_state.get("active_vehicles", [])}
    plate = random.choice([p for p in PLATES if p not in parked] or PLATES)
    time_in = datetime.now()
    if overnight:
        time_in -= timedelta(days=1, hours=random.randint(1, 5))
    return make_vehicle(plate, vtype, time_in)


def is_overnight(v):
    # Research flow: no record for the current date = vehicle overnighted
    return v["time_in"].date() < datetime.now().date()


def compute_bill(v, at=None):
    at = at or datetime.now()
    duration = (at - v["time_in"]).total_seconds() / 3600
    base = max(1, int(np.ceil(duration))) * v["rate"]
    violation = VIOLATION_FEE if v["time_in"].date() < at.date() else 0
    return duration, base, violation, base + violation


def peso(x):
    return f"₱{x:,.2f}"


# =====================================================================
# 3. SESSION STATE
# =====================================================================
ss = st.session_state
if "active_vehicles" not in ss:
    n = datetime.now()
    ss.active_vehicles = [
        make_vehicle("INIT-001", "SUV", n - timedelta(hours=2, minutes=15)),
        make_vehicle("INIT-002", "Sedan", n - timedelta(minutes=45)),
        make_vehicle("INIT-003", "Motorcycle", n - timedelta(minutes=10)),
    ]
for v in ss.active_vehicles:  # keep older sessions working
    v.setdefault("model", random.choice(MODELS[v["type"]]))
    v.setdefault("confidence", round(random.uniform(0.88, 0.99), 3))
    v.setdefault("response_ms", random.randint(180, 450))

if "history7" not in ss:  # vehicles per day, last 7 days
    days = [datetime.now() - timedelta(days=i) for i in range(6, -1, -1)]
    hist = pd.DataFrame(0, index=[d.strftime("%b %d") for d in days], columns=list(VEHICLE_RATES))
    for d in days:
        cap = 6 if d.weekday() < 5 else 3
        for t in VEHICLE_RATES:
            hist.loc[d.strftime("%b %d"), t] = random.randint(0, cap)
    ss.history7 = hist

if "log" not in ss:  # completed sessions (seeded so the dashboard isn't empty)
    ss.log = []
    n = datetime.now()
    for plate, vtype, ago, mins in [("SEED-101", "Sedan", 5, 70), ("SEED-102", "Van", 4, 95), ("SEED-103", "Hatchback", 3, 50)]:
        v = make_vehicle(plate, vtype, n - timedelta(hours=ago))
        out = v["time_in"] + timedelta(minutes=mins)
        dur, base, vio, total = compute_bill(v, out)
        ss.log.append({**v, "time_out": out, "duration": dur, "base": base, "violation": vio, "total": total})

if "detections" not in ss:
    ss.detections = [{"confidence": v["confidence"], "response_ms": v["response_ms"]} for v in ss.active_vehicles]
if "pending_exit" not in ss:
    ss.pending_exit = None


def register(v):
    ss.active_vehicles.append(v)
    ss.detections.append({"confidence": v["confidence"], "response_ms": v["response_ms"]})
    label = v["time_in"].strftime("%b %d")
    if label in ss.history7.index:
        ss.history7.loc[label, v["type"]] += 1


# =====================================================================
# 4. SIDEBAR
# =====================================================================
PAGES = ["📊 Overview", "🚘 Live Sessions", "🚪 Exit & Billing", "🧾 Session Log", "⚙️ Rates"]

if st.sidebar.button("+ Simulate Detection", type="primary"):
    nv = generate_mock_vehicle()
    register(nv)
    st.toast(f"Gate opened: {nv['model']} classified as {nv['type']} ({nv['plate']})", icon="🚗")
    st.rerun()
if st.sidebar.button("🌙 Simulate Overnight Vehicle"):
    nv = generate_mock_vehicle(overnight=True)
    register(nv)
    st.toast(f"{nv['type']} ({nv['plate']}) has no record for today: violation fee applies.", icon="⚠️")
    st.rerun()
st.sidebar.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

# =====================================================================
# 5. SHARED PIECES
# =====================================================================
def svg(inner):
    return ('<svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="#2563EB" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


SVG_CAR = svg('<path d="M3 13l2-6a2 2 0 0 1 2-1.5h10a2 2 0 0 1 2 1.5l2 6"/>'
              '<path d="M3 13h18v5a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-1H7v1a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>'
              '<circle cx="7.5" cy="15.5" r=".7"/><circle cx="16.5" cy="15.5" r=".7"/>')
SVG_PESO = svg('<circle cx="12" cy="12" r="9"/><text x="12" y="16.5" text-anchor="middle" font-size="12" '
               'font-weight="700" fill="#2563EB" stroke="none">₱</text>')
SVG_EXIT = svg('<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/>'
               '<line x1="21" y1="12" x2="9" y2="12"/>')
SVG_WARN = svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
               '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>')


def stat_card(icon, value, label):
    return f'<div class="stat">{icon}<div><div class="num">{value}</div><div class="lbl">{label}</div></div></div>'


def card(height=None):
    if height is None:
        return st.container(border=True)
    try:
        return st.container(border=True, height=height)
    except TypeError:  # older Streamlit without fixed-height containers
        return st.container(border=True)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def page_header(title):
    det = ss.detections
    conf = np.mean([d["confidence"] for d in det]) * 100 if det else 0
    resp = np.mean([d["response_ms"] for d in det]) if det else 0
    left, right = st.columns([3, 1.4])
    left.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    right.markdown(
        f'<div class="status"><span class="dot"></span><div><div class="s-title">Entry Gate · Operational</div>'
        f'<div class="s-sub">Camera classifying · {conf:.1f}% confidence · {resp:.0f} ms (simulated)</div></div></div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def active_table():
    rows = []
    for v in ss.active_vehicles:
        dur, base, vio, total = compute_bill(v)
        rows.append({
            "Plate No.": v["plate"], "Model": v["model"], "Vehicle Type": v["type"],
            "Time In": v["time_in"].strftime("%b %d %H:%M:%S"), "Time Out": "— (parked)",
            "Duration (hrs)": round(dur, 2), "Hourly Rate": f"₱{v['rate']}",
            "Violation Fee": peso(vio), "Current Bill": peso(total),
            "Status": "⚠️ Overnight" if vio else "🟢 Parked",
        })
    return pd.DataFrame(rows)


# =====================================================================
# 6. PAGES
# =====================================================================
st.markdown(
    f'<div class="topbar"><div class="brand">SMART<span>PARK</span><em>IoT Access &amp; Billing</em></div>'
    f'<div class="topmeta">{datetime.now():%A, %b %d, %Y}</div></div>', unsafe_allow_html=True)


def overview():
    page_header("Overview")
    active = ss.active_vehicles
    collected = sum(e["total"] for e in ss.log)
    pending = sum(compute_bill(v)[3] for v in active)
    over = sum(1 for v in active if is_overnight(v))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card(SVG_CAR, len(active), "Active vehicles"), unsafe_allow_html=True)
    c2.markdown(stat_card(SVG_PESO, peso(collected), "Revenue collected"), unsafe_allow_html=True)
    c3.markdown(stat_card(SVG_EXIT, peso(pending), "Pending bills (parked)"), unsafe_allow_html=True)
    c4.markdown(stat_card(SVG_WARN, over, "Overnight violations"), unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ---- row 1: traffic chart + top vehicle types ----
    left, right = st.columns([1.35, 1])
    hist = ss.history7
    with left:
        with card(420):
            h1, h2 = st.columns([2, 1.1])
            h1.markdown('<div class="card-title">Daily vehicle traffic</div>', unsafe_allow_html=True)
            choice = h2.selectbox("Vehicle type", ["All vehicles"] + list(VEHICLE_RATES),
                                  label_visibility="collapsed", key="traffic_filter")
            series = hist.sum(axis=1) if choice == "All vehicles" else hist[choice]
            st.markdown(
                f'<div class="big">Last 7 days:<b>{int(series.sum())}</b>Today:<b class="gray">{int(series.iloc[-1])}</b></div>',
                unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(
                x=list(series.index), y=series.values, mode="lines",
                line=dict(color="#2563EB", width=3, shape="spline", smoothing=1.0),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.14)",
                hovertemplate="%{x}: %{y} vehicles<extra></extra>"))
            fig.update_layout(height=290, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white", plot_bgcolor="white",
                              showlegend=False, font=dict(color="#64748B", size=12))
            fig.update_xaxes(showgrid=False, showline=True, linecolor="#E2E8F0")
            fig.update_yaxes(rangemode="tozero", gridcolor="#EEF2F7", zeroline=False)
            st.plotly_chart(fig, config={"displaylogo": False})
    with right:
        with card(420):
            st.markdown('<div class="card-title">Top vehicle types</div>', unsafe_allow_html=True)
            top = hist.sum().sort_values(ascending=False).head(4)
            html = "".join(
                f'<div class="row"><div class="av">{ICONS[t]}</div><div class="mid"><div class="name">{t}</div>'
                f'<div class="sub">Total entries: {int(n)}</div></div><div class="badge">₱{VEHICLE_RATES[t]}/hr</div></div>'
                for t, n in top.items())
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ---- row 2: session donut + latest sessions ----
    left, right = st.columns([1.35, 1])
    with left:
        with card(400):
            st.markdown('<div class="card-title">Session report</div>', unsafe_allow_html=True)
            parked = len(active) - over
            vals = [parked, over, len(ss.log)]
            if sum(vals) == 0:
                st.markdown('<div class="empty">No sessions yet.</div>', unsafe_allow_html=True)
            else:
                fig = go.Figure(go.Pie(
                    labels=["Parked", "Overnight", "Exited"], values=vals, hole=0.68, sort=False, textinfo="none",
                    marker=dict(colors=["#2563EB", "#F59E0B", "#CBD5E1"], line=dict(color="white", width=3)),
                    domain=dict(x=[0.38, 1]), hovertemplate="%{label}: %{value}<extra></extra>"))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white",
                    legend=dict(x=0.0, y=0.5, xanchor="left", yanchor="middle", font=dict(size=13, color="#334155")),
                    annotations=[dict(text=f"<b>{sum(vals)}</b><br><span style='font-size:12px;color:#94A3B8'>sessions</span>",
                                      x=0.69, y=0.5, showarrow=False, font=dict(size=24, color="#0F172A"))])
                st.plotly_chart(fig, config={"displayModeBar": False})
    with right:
        with card(400):
            st.markdown('<div class="card-title">Latest sessions</div>', unsafe_allow_html=True)
            events = []
            for v in active:
                _, _, vio, total = compute_bill(v)
                events.append((v["time_in"], v, "Overnight" if vio else "Parked", "s-over" if vio else "s-parked", total))
            for e in ss.log:
                events.append((e["time_out"], e, "Exited", "s-exit", e["total"]))
            events.sort(key=lambda x: x[0], reverse=True)
            html = "".join(
                f'<div class="row"><div class="av">{ICONS[v["type"]]}</div><div class="mid"><div class="name">{v["plate"]}</div>'
                f'<div class="sub">{v["model"]} · {v["type"]}</div></div>'
                f'<div class="right"><div class="{cls}">{status}</div><div class="amt">{peso(total)}</div></div></div>'
                for _, v, status, cls, total in events[:4])
            st.markdown(html or '<div class="empty">No sessions yet.</div>', unsafe_allow_html=True)


def sessions_page():
    page_header("Live Sessions")
    with card():
        st.markdown('<div class="card-title">Active parking sessions</div>', unsafe_allow_html=True)
        if ss.active_vehicles:
            show_df(active_table())
        else:
            st.markdown('<div class="empty">No vehicles are currently parked.</div>', unsafe_allow_html=True)


def exit_page():
    page_header("Exit & Billing")
    st.caption("Exit flow: scan plate → find record → compute fee → operator confirms payment → barrier opens")
    with card():
        if ss.pending_exit is None:
            st.markdown('<div class="card-title">Exit camera</div>', unsafe_allow_html=True)
            if not ss.active_vehicles:
                st.markdown('<div class="empty">No vehicles currently parked.</div>', unsafe_allow_html=True)
                return
            pick = st.selectbox("Select vehicle at exit camera:",
                                [f"{v['plate']} ({v['type']})" for v in ss.active_vehicles])
            if st.button("📷 Scan Plate & Compute Fee", type="primary"):
                plate = pick.split(" ")[0]
                vd = next(v for v in ss.active_vehicles if v["plate"] == plate)
                t_out = datetime.now()
                dur, base, vio, total = compute_bill(vd, t_out)
                ss.pending_exit = {"plate": plate, "time_out": t_out, "duration": dur, "base": base, "violation": vio, "total": total}
                st.rerun()
            return

        p = ss.pending_exit
        vd = next((v for v in ss.active_vehicles if v["plate"] == p["plate"]), None)
        if vd is None:
            ss.pending_exit = None
            st.rerun()

        info_col = st.container()
        with info_col:
            if p["violation"]:
                st.error("⚠️ No record for today: vehicle overnighted. Violation fee applied.")
            else:
                st.success("✅ Record found for today.")
            st.markdown(
                f"**Plate:** {vd['plate']}  \n**Model / Type:** {vd['model']} / {vd['type']}  \n"
                f"**Time In:** {vd['time_in']:%b %d %H:%M:%S}  \n**Time Out:** {p['time_out']:%b %d %H:%M:%S}  \n"
                f"**Duration:** {p['duration']:.2f} hrs  \n**Parking Fee:** {peso(p['base'])}  \n"
                f"**Violation Fee:** {peso(p['violation'])}")
            st.metric("Total Amount Due", peso(p["total"]))
            b1, b2 = st.columns(2)
            if b1.button("✅ Confirm Payment & Open Barrier", type="primary"):
                ss.log.append({**vd, "time_out": p["time_out"], "duration": p["duration"],
                               "base": p["base"], "violation": p["violation"], "total": p["total"]})
                ss.active_vehicles.remove(vd)
                ss.pending_exit = None
                st.toast(f"Payment received: {peso(p['total'])}. Barrier open, vehicle exited.", icon="✅")
                st.rerun()
            if b2.button("✖ Cancel"):
                ss.pending_exit = None
                st.rerun()


def log_page():
    page_header("Session Log")
    with card():
        st.markdown('<div class="card-title">Completed sessions (time in / time out)</div>', unsafe_allow_html=True)
        if ss.log:
            rows = [{
                "Plate No.": e["plate"], "Model": e["model"], "Vehicle Type": e["type"],
                "Time In": e["time_in"].strftime("%b %d %H:%M:%S"), "Time Out": e["time_out"].strftime("%b %d %H:%M:%S"),
                "Duration (hrs)": round(e["duration"], 2), "Violation Fee": peso(e["violation"]), "Amount Paid": peso(e["total"]),
            } for e in ss.log[::-1]]
            show_df(pd.DataFrame(rows))
        else:
            st.markdown('<div class="empty">No vehicles have exited yet.</div>', unsafe_allow_html=True)


def rates_page():
    page_header("Rates")
    left, right = st.columns([1.35, 1])
    with left:
        with card():
            st.markdown('<div class="card-title">Hourly rate by vehicle type</div>', unsafe_allow_html=True)
            show_df(pd.DataFrame([{"Vehicle Type": f"{ICONS[t]} {t}", "Rate per hour": f"₱{r}"} for t, r in VEHICLE_RATES.items()]))
    with right:
        with card():
            st.markdown('<div class="card-title">Billing rules</div>', unsafe_allow_html=True)
            st.metric("Overnight violation fee", peso(VIOLATION_FEE))
            st.markdown(
                "- Billed per started hour (minimum 1 hour)\n"
                "- Rate depends on the classified vehicle type\n"
                "- No record for today means overnighted: violation fee is added")


{PAGES[0]: overview, PAGES[1]: sessions_page, PAGES[2]: exit_page, PAGES[3]: log_page, PAGES[4]: rates_page}[page]()