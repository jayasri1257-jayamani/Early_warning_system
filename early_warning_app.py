"""
Early Warning System for Academically At-Risk Students
------------------------------------------------------
Pure Python website built with Streamlit.

Run:
    pip install streamlit pandas numpy
    streamlit run early_warning_app.py

Everything you may want to edit is grouped at the top (USERS, FACTORS,
THRESHOLDS, RECOMMENDATIONS). Risk scores are also adjustable live from the
"Risk Settings" page once you log in as admin or teacher.
"""

import copy
import datetime as dt
import os

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# 1. CONFIGURATION  (edit freely)
# ----------------------------------------------------------------------------
APP_TITLE = "🎓 Early Warning System – Academic Risk"
INTERVENTION_FILE = "interventions.csv"   # saved next to this script
STUDENT_DEFAULT_PASSWORD = "student123"   # students log in with their Student ID

# Demo staff accounts. CHANGE THESE before real use (or connect a database).
USERS = {
    "admin":    {"password": "admin123",   "role": "Administrator", "name": "College Admin"},
    "teacher1": {"password": "teacher123", "role": "Teacher",       "name": "Prof. Rao"},
    "mentor1":  {"password": "mentor123",  "role": "Mentor",        "name": "Dr. Meena"},  # sees students whose 'mentor' == this name
}

# Each factor: label, target, weight, higher_is_better
#  - higher_is_better=True : risk grows as the value falls below `target`
#  - higher_is_better=False: `target` is the value at which risk is 100% (e.g. 3 backlogs)
FACTORS = {
    "attendance":            {"label": "Attendance (%)",            "target": 85, "weight": 25, "higher_is_better": True},
    "internal_marks":        {"label": "Internal marks (%)",        "target": 60, "weight": 25, "higher_is_better": True},
    "assignment_completion": {"label": "Assignment completion (%)", "target": 80, "weight": 15, "higher_is_better": True},
    "prev_gpa":              {"label": "Previous GPA (/10)",        "target": 7,  "weight": 15, "higher_is_better": True},
    "backlogs":              {"label": "Backlogs (count)",          "target": 3,  "weight": 10, "higher_is_better": False},
    "study_hours_week":      {"label": "Study hours / week",        "target": 15, "weight": 5,  "higher_is_better": True},
    "participation":         {"label": "Class participation (%)",   "target": 60, "weight": 5,  "higher_is_better": True},
}

THRESHOLDS = {"high": 35, "medium": 18}   # risk score (0-100) cut-offs
FACTOR_ALERT_LEVEL = 30                    # a factor is a "reason" if its own risk >= this

RECOMMENDATIONS = {
    "attendance": "Meet the student to understand absences; set an attendance recovery plan and inform parents/guardian if needed.",
    "internal_marks": "Arrange remedial classes or peer tutoring; share a topic-wise revision plan before finals.",
    "assignment_completion": "Give a short extension window with weekly check-ins on pending assignments.",
    "prev_gpa": "Assign a mentor for study-skills coaching and review of weak subjects from earlier semesters.",
    "backlogs": "Plan a backlog-clearing schedule and point the student to supplementary exam options.",
    "study_hours_week": "Help build a weekly timetable; suggest study groups or library sessions.",
    "participation": "Encourage participation with small in-class tasks; check for confidence or personal issues.",
}

REQUIRED_COLUMNS = ["student_id", "name", "department", "semester", "mentor"] + list(FACTORS.keys())


# ----------------------------------------------------------------------------
# 2. DATA (sample generator + loaders)
# ----------------------------------------------------------------------------
def generate_sample_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    first = ["Aarav", "Diya", "Kabir", "Ananya", "Rohan", "Meera", "Vihaan", "Isha", "Arjun", "Priya",
             "Karthik", "Sneha", "Rahul", "Lakshmi", "Suresh", "Divya", "Naveen", "Pooja", "Vikram", "Nisha"]
    last = ["Kumar", "Sharma", "Reddy", "Nair", "Iyer", "Patel", "Singh", "Das", "Menon", "Gupta"]
    mentors = ["Dr. Meena", "Dr. Suresh", "Prof. Anitha", "Dr. Ravi"]
    depts = ["CSE", "ECE", "MECH", "CIVIL", "IT"]

    ability = rng.normal(0, 1, n)  # hidden factor so the columns correlate realistically
    clip = np.clip
    df = pd.DataFrame({
        "student_id": [f"S{1001 + i}" for i in range(n)],
        "name": [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)],
        "department": rng.choice(depts, n),
        "semester": rng.integers(1, 9, n),
        "mentor": rng.choice(mentors, n),
        "attendance": clip(78 + 8 * ability + rng.normal(0, 8, n), 35, 100).round(1),
        "internal_marks": clip(60 + 12 * ability + rng.normal(0, 7, n), 15, 100).round(1),
        "assignment_completion": clip(75 + 10 * ability + rng.normal(0, 10, n), 10, 100).round(1),
        "prev_gpa": clip(7 + 0.9 * ability + rng.normal(0, 0.4, n), 3, 10).round(2),
        "backlogs": clip(np.round(np.maximum(0, -ability + rng.normal(0, 0.7, n))), 0, 6).astype(int),
        "study_hours_week": clip(12 + 3 * ability + rng.normal(0, 3, n), 1, 30).round(1),
        "participation": clip(60 + 12 * ability + rng.normal(0, 10, n), 10, 100).round(1),
    })
    return df


def load_interventions() -> pd.DataFrame:
    cols = ["student_id", "date", "by", "action", "note"]
    if os.path.exists(INTERVENTION_FILE):
        return pd.read_csv(INTERVENTION_FILE)
    return pd.DataFrame(columns=cols)


def save_intervention(student_id, by, action, note):
    df = load_interventions()
    new = pd.DataFrame([{"student_id": student_id, "date": dt.date.today().isoformat(),
                         "by": by, "action": action, "note": note}])
    pd.concat([df, new], ignore_index=True).to_csv(INTERVENTION_FILE, index=False)


# ----------------------------------------------------------------------------
# 3. RISK ENGINE
# ----------------------------------------------------------------------------
def factor_risk(value: float, cfg: dict) -> float:
    """Return risk 0-100 for one factor."""
    if cfg["higher_is_better"]:
        risk = (cfg["target"] - value) / cfg["target"] * 100
    else:
        risk = value / cfg["target"] * 100
    return float(np.clip(risk, 0, 100))


def score_students(df: pd.DataFrame, factors: dict, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    total_w = sum(f["weight"] for f in factors.values()) or 1
    risk_cols = {}
    for col, cfg in factors.items():
        out[f"risk_{col}"] = out[col].apply(lambda v, c=cfg: factor_risk(v, c))
        risk_cols[col] = f"risk_{col}"
    out["risk_score"] = sum(out[rc] * factors[c]["weight"] for c, rc in risk_cols.items()) / total_w
    out["risk_score"] = out["risk_score"].round(1)
    out["risk_level"] = np.where(out["risk_score"] >= thresholds["high"], "High",
                         np.where(out["risk_score"] >= thresholds["medium"], "Medium", "Low"))
    out["reasons"] = out.apply(
        lambda r: [factors[c]["label"] for c in factors if r[f"risk_{c}"] >= FACTOR_ALERT_LEVEL], axis=1)
    return out


def recommendations_for(row: pd.Series, factors: dict) -> list:
    recs = [RECOMMENDATIONS[c] for c in factors if row[f"risk_{c}"] >= FACTOR_ALERT_LEVEL and c in RECOMMENDATIONS]
    return recs or ["No major concerns. Keep encouraging the current routine."]


LEVEL_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


# ----------------------------------------------------------------------------
# 4. STATE + LOGIN
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("df", generate_sample_data())
    st.session_state.setdefault("factors", copy.deepcopy(FACTORS))
    st.session_state.setdefault("thresholds", dict(THRESHOLDS))
    st.session_state.setdefault("exam_date", dt.date.today() + dt.timedelta(days=45))
    st.session_state.setdefault("user", None)


def login_screen():
    st.title(APP_TITLE)
    st.caption("Detect students who need academic support before final exams.")
    with st.form("login"):
        username = st.text_input("Username / Student ID")
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.user = {"id": username, **user}
            st.rerun()
        elif username in set(st.session_state.df["student_id"]) and password == STUDENT_DEFAULT_PASSWORD:
            name = st.session_state.df.loc[st.session_state.df.student_id == username, "name"].iloc[0]
            st.session_state.user = {"id": username, "role": "Student", "name": name}
            st.rerun()
        else:
            st.error("Invalid credentials.")
    with st.expander("Demo logins"):
        st.write("**admin / admin123**, **teacher1 / teacher123**, **mentor1 / mentor123**")
        st.write(f"Student: any Student ID (e.g. **S1001**) with password **{STUDENT_DEFAULT_PASSWORD}**")


# ----------------------------------------------------------------------------
# 5. PAGES
# ----------------------------------------------------------------------------
def visible_data(scored: pd.DataFrame) -> pd.DataFrame:
    """Role-based filtering."""
    user = st.session_state.user
    if user["role"] == "Mentor":
        return scored[scored["mentor"] == user["name"]]
    if user["role"] == "Student":
        return scored[scored["student_id"] == user["id"]]
    return scored


def page_overview(scored):
    st.header("📊 Overview")
    days_left = (st.session_state.exam_date - dt.date.today()).days
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", len(scored))
    c2.metric("High risk", int((scored.risk_level == "High").sum()))
    c3.metric("Medium risk", int((scored.risk_level == "Medium").sum()))
    c4.metric("Avg attendance", f"{scored.attendance.mean():.1f}%")
    c5.metric("Days to finals", max(days_left, 0))

    if scored.empty:
        st.info("No students to show.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Risk level distribution")
        st.bar_chart(scored["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0))
    with right:
        st.subheader("Average risk by department")
        st.bar_chart(scored.groupby("department")["risk_score"].mean().round(1))

    st.subheader("Most common risk reasons")
    reasons = pd.Series([r for lst in scored[scored.risk_level != "Low"]["reasons"] for r in lst])
    if reasons.empty:
        st.write("None 🎉")
    else:
        st.bar_chart(reasons.value_counts())


def page_at_risk(scored):
    st.header("🚨 At-Risk Students")
    c1, c2, c3 = st.columns(3)
    levels = c1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    dept = c2.multiselect("Department", sorted(scored.department.unique()))
    search = c3.text_input("Search name / ID")

    view = scored[scored.risk_level.isin(levels)]
    if dept:
        view = view[view.department.isin(dept)]
    if search:
        s = search.lower()
        view = view[view.name.str.lower().str.contains(s) | view.student_id.str.lower().str.contains(s)]
    view = view.sort_values("risk_score", ascending=False)

    table = view[["student_id", "name", "department", "semester", "mentor", "attendance",
                  "internal_marks", "risk_score", "risk_level", "reasons"]].copy()
    table["risk_level"] = table["risk_level"].map(LEVEL_ICON)
    table["reasons"] = table["reasons"].apply(", ".join)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download this list (CSV)", table.to_csv(index=False), "at_risk_students.csv")


def page_student_detail(scored):
    st.header("🧑‍🎓 Student Detail")
    if scored.empty:
        st.info("No students available.")
        return
    options = scored.sort_values("risk_score", ascending=False)
    labels = {r.student_id: f"{r.student_id} – {r['name']}" for _, r in options.iterrows()}
    sid = st.selectbox("Select student", list(labels), format_func=labels.get)
    row = scored[scored.student_id == sid].iloc[0]
    factors = st.session_state.factors

    st.subheader(f"{row['name']}  ({LEVEL_ICON[row.risk_level]})")
    st.caption(f"{row.department} • Semester {row.semester} • Mentor: {row.mentor}")
    st.progress(min(int(row.risk_score), 100), text=f"Risk score: {row.risk_score}/100")

    cols = st.columns(len(factors))
    for col, (key, cfg) in zip(cols, factors.items()):
        col.metric(cfg["label"], row[key])

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk contribution by factor**")
        chart = pd.Series({cfg["label"]: row[f"risk_{k}"] for k, cfg in factors.items()})
        st.bar_chart(chart)
    with right:
        st.markdown("**Why flagged**")
        if row.reasons:
            for r in row.reasons:
                st.write(f"• {r}")
        else:
            st.write("No factor above alert level.")
        st.markdown("**Suggested actions**")
        for rec in recommendations_for(row, factors):
            st.write(f"✅ {rec}")

    st.markdown("---")
    st.markdown("**Support history**")
    hist = load_interventions()
    hist = hist[hist.student_id == sid]
    if hist.empty:
        st.write("No interventions recorded yet.")
    else:
        st.dataframe(hist.sort_values("date", ascending=False), hide_index=True, use_container_width=True)


def page_interventions(scored):
    st.header("📝 Record Support / Intervention")
    labels = {r.student_id: f"{r.student_id} – {r['name']} ({r.risk_level})" for _, r in scored.iterrows()}
    with st.form("intervention"):
        sid = st.selectbox("Student", list(labels), format_func=labels.get)
        action = st.selectbox("Action", ["Counselling meeting", "Remedial class", "Parent contacted",
                                         "Peer tutoring", "Study plan created", "Attendance warning", "Other"])
        note = st.text_area("Notes")
        if st.form_submit_button("Save"):
            save_intervention(sid, st.session_state.user["name"], action, note)
            st.success("Saved.")
    st.subheader("All records")
    hist = load_interventions()
    hist = hist[hist.student_id.isin(scored.student_id)]
    st.dataframe(hist.sort_values("date", ascending=False), hide_index=True, use_container_width=True)


def page_data():
    st.header("📁 Data")
    st.write("Upload a CSV with these columns (one row per student):")
    st.code(", ".join(REQUIRED_COLUMNS))
    template = generate_sample_data(5)
    st.download_button("⬇️ Download template / sample CSV", template.to_csv(index=False), "students_template.csv")
    file = st.file_uploader("Upload student data (CSV)", type="csv")
    if file:
        new = pd.read_csv(file)
        missing = [c for c in REQUIRED_COLUMNS if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.session_state.df = new[REQUIRED_COLUMNS]
            st.success(f"Loaded {len(new)} students.")
    if st.button("Reset to sample data"):
        st.session_state.df = generate_sample_data()
        st.rerun()
    st.subheader("Current data")
    st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)


def page_settings():
    st.header("⚙️ Risk Settings")
    st.caption("Tune how risk is calculated. Changes apply instantly (until the app restarts).")
    st.session_state.exam_date = st.date_input("Final exam start date", st.session_state.exam_date)
    th = st.session_state.thresholds
    c1, c2 = st.columns(2)
    th["high"] = c1.slider("High-risk score from", 0, 100, th["high"])
    th["medium"] = c2.slider("Medium-risk score from", 0, 100, min(th["medium"], th["high"]))
    st.markdown("**Factor targets & weights**")
    for key, cfg in st.session_state.factors.items():
        a, b = st.columns(2)
        cfg["target"] = a.number_input(f"{cfg['label']} – target", value=float(cfg["target"]), key=f"t_{key}")
        cfg["weight"] = b.number_input(f"{cfg['label']} – weight", value=float(cfg["weight"]), min_value=0.0, key=f"w_{key}")


def page_my_status(scored):
    st.header("🙋 My Academic Status")
    if scored.empty:
        st.info("Your record was not found.")
        return
    row = scored.iloc[0]
    factors = st.session_state.factors
    st.subheader(f"Hello, {row['name']}")
    st.metric("Current risk level", LEVEL_ICON[row.risk_level])
    st.progress(min(int(row.risk_score), 100), text=f"Risk score: {row.risk_score}/100")
    cols = st.columns(len(factors))
    for col, (key, cfg) in zip(cols, factors.items()):
        col.metric(cfg["label"], row[key], help=f"Target: {cfg['target']}")
    st.markdown("**Where you can improve**")
    for rec in recommendations_for(row, factors):
        st.write(f"✅ {rec}")
    st.info(f"Your mentor: **{row.mentor}**. Reach out early – support is available before finals!")


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Early Warning System", page_icon="🎓", layout="wide")
    init_state()

    if not st.session_state.user:
        login_screen()
        return

    user = st.session_state.user
    scored_all = score_students(st.session_state.df, st.session_state.factors, st.session_state.thresholds)
    scored = visible_data(scored_all)

    st.sidebar.title("🎓 EWS")
    st.sidebar.write(f"**{user['name']}**  \n{user['role']}")

    pages = {
        "Administrator": ["Overview", "At-Risk Students", "Student Detail", "Interventions", "Data", "Risk Settings"],
        "Teacher":       ["Overview", "At-Risk Students", "Student Detail", "Interventions", "Data", "Risk Settings"],
        "Mentor":        ["Overview", "At-Risk Students", "Student Detail", "Interventions"],
        "Student":       ["My Status"],
    }[user["role"]]
    choice = st.sidebar.radio("Go to", pages)
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    if choice == "Overview":
        page_overview(scored)
    elif choice == "At-Risk Students":
        page_at_risk(scored)
    elif choice == "Student Detail":
        page_student_detail(scored)
    elif choice == "Interventions":
        page_interventions(scored)
    elif choice == "Data":
        page_data()
    elif choice == "Risk Settings":
        page_settings()
    elif choice == "My Status":
        page_my_status(scored)


if __name__ == "__main__":
    main()
