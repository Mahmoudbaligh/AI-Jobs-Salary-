"""
Global AI Jobs — Salary Predictor & EDA Dashboard
Run with: streamlit run ai_salary_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Jobs Salary Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card h2 { color: #5b9cf6; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #9ba8c2; font-size: 0.9rem; margin: 4px 0 0; }
    .prediction-box {
        background: linear-gradient(135deg, #1a2a4a, #1e3058);
        border: 2px solid #4a7eda;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .prediction-box h1 { color: #6fb3f9; font-size: 3.5rem; margin: 0; }
    .prediction-box p  { color: #aac4e8; font-size: 1.1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #161a28; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD & CACHE DATA ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try local path first, then kaggle path
    for path in ["global_ai_jobs.csv", "D:\\AI\\AI Salary\\global_ai_jobs.csv"]:
        try:
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            continue
    st.error("❌ Could not find global_ai_jobs.csv. Place it in the same folder as this script.")
    st.stop()

@st.cache_resource
def train_model(df):
    """Train model using numeric + encoded categorical columns."""
    le_dict = {}
    df_model = df.copy()

    cat_cols = ["job_role", "country", "experience_level", "work_mode",
                "company_size", "education_required", "industry"]
    for col in cat_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le

    feature_cols = (
        [c + "_enc" for c in cat_cols if c in df_model.columns]
        + ["experience_years", "weekly_hours", "company_rating",
           "ai_adoption_score", "cost_of_living_index",
           "skill_demand_score", "job_security_score",
           "career_growth_score", "economic_index"]
    )
    feature_cols = [c for c in feature_cols if c in df_model.columns]

    X = df_model[feature_cols]
    y = df_model["salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42,
                                  max_depth=12, min_samples_leaf=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2":  round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 0),
    }
    return model, le_dict, feature_cols, metrics, X_test, y_test, y_pred

df = load_data()
model, le_dict, feature_cols, model_metrics, X_test, y_test, y_pred = train_model(df)

# ─── SIDEBAR — PREDICTION INPUTS ─────────────────────────────────────────────
with st.sidebar:
    st.image("cover.webp", width=250)
    st.title("Salary Predictor")
    st.markdown("---")
    st.subheader("Your Profile")

    job_role = st.selectbox("💼 Job Title", sorted(df["job_role"].unique()))
    country  = st.selectbox("🌍 Country",   sorted(df["country"].unique()))
    exp_yrs  = st.slider("📅 Years of Experience", 0, 20, 5)

    exp_level_options = ["Entry", "Mid", "Senior", "Lead"]
    exp_level = st.select_slider("🎯 Experience Level", options=exp_level_options,
                                 value="Mid")

    st.markdown("---")
    st.subheader("More Details (Optional)")
    work_mode   = st.selectbox("🏠 Work Mode",     sorted(df["work_mode"].unique()))
    company_sz  = st.selectbox("🏢 Company Size",  sorted(df["company_size"].unique()))
    education   = st.selectbox("🎓 Education",     sorted(df["education_required"].unique()))
    industry    = st.selectbox("🏭 Industry",      sorted(df["industry"].unique()))
    weekly_hrs  = st.slider("⏰ Weekly Hours", 30, 60, 40)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict My Salary", use_container_width=True, type="primary")

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict_salary(job_role, country, exp_yrs, exp_level, work_mode,
                   company_sz, education, industry, weekly_hrs):
    # Averages from dataset for optional numeric features
    defaults = df.groupby(["country", "job_role"])[
        ["company_rating", "ai_adoption_score", "cost_of_living_index",
         "skill_demand_score", "job_security_score", "career_growth_score",
         "economic_index"]
    ].mean()

    try:
        defaults_row = defaults.loc[(country, job_role)]
    except KeyError:
        defaults_row = df[["company_rating", "ai_adoption_score", "cost_of_living_index",
                           "skill_demand_score", "job_security_score", "career_growth_score",
                           "economic_index"]].mean()

    row = {}
    enc_map = {
        "job_role": job_role, "country": country, "experience_level": exp_level,
        "work_mode": work_mode, "company_size": company_sz,
        "education_required": education, "industry": industry,
    }
    for col, val in enc_map.items():
        if col + "_enc" in feature_cols:
            le = le_dict[col]
            try:
                row[col + "_enc"] = le.transform([val])[0]
            except ValueError:
                row[col + "_enc"] = 0

    row["experience_years"] = exp_yrs
    row["weekly_hours"]     = weekly_hrs
    row["company_rating"]   = defaults_row.get("company_rating", df["company_rating"].mean())
    row["ai_adoption_score"]     = defaults_row.get("ai_adoption_score", df["ai_adoption_score"].mean())
    row["cost_of_living_index"]  = defaults_row.get("cost_of_living_index", df["cost_of_living_index"].mean())
    row["skill_demand_score"]    = defaults_row.get("skill_demand_score", df["skill_demand_score"].mean())
    row["job_security_score"]    = defaults_row.get("job_security_score", df["job_security_score"].mean())
    row["career_growth_score"]   = defaults_row.get("career_growth_score", df["career_growth_score"].mean())
    row["economic_index"]        = defaults_row.get("economic_index", df["economic_index"].mean())

    X_input = pd.DataFrame([row])[feature_cols]
    return model.predict(X_input)[0]

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.title(" Global AI Jobs — Salary Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 EDA & Insights",
                                   "🤖 Model Performance", "🗺️ Country Comparison"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_info = st.columns([1.2, 1], gap="large")

    with col_pred:
        st.subheader(" Salary Prediction")

        if predict_btn or True:   # also show on load with defaults
            pred = predict_salary(job_role, country, exp_yrs, exp_level,
                                  work_mode, company_sz, education, industry, weekly_hrs)

            st.markdown(f"""
            <div class="prediction-box">
                <p>Estimated Annual Salary</p>
                <h1>${pred:,.0f}</h1>
                <p>USD · {job_role} · {country} · {exp_yrs} yrs experience</p>
            </div>
            """, unsafe_allow_html=True)

            # Contextual metrics
            st.markdown("&nbsp;")
            m1, m2, m3 = st.columns(3)
            role_avg   = df[df["job_role"] == job_role]["salary_usd"].mean()
            country_avg = df[df["country"] == country]["salary_usd"].mean()
            global_avg  = df["salary_usd"].mean()

            m1.metric("Role Average",   f"${role_avg:,.0f}",
                      f"{(pred-role_avg)/role_avg*100:+.1f}%")
            m2.metric("Country Average", f"${country_avg:,.0f}",
                      f"{(pred-country_avg)/country_avg*100:+.1f}%")
            m3.metric("Global Average",  f"${global_avg:,.0f}",
                      f"{(pred-global_avg)/global_avg*100:+.1f}%")

            # Salary range for role+country
            st.markdown("#### Salary Range for Your Profile")
            subset = df[(df["job_role"] == job_role) & (df["country"] == country)]["salary_usd"]
            if len(subset) > 10:
                fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="#0e1117")
                ax.set_facecolor("#161a28")
                ax.hist(subset, bins=30, color="#4a7eda", alpha=0.8, edgecolor="#2a3a6a")
                ax.axvline(pred, color="#f5a623", lw=2.5, label=f"Your prediction: ${pred:,.0f}")
                ax.axvline(subset.mean(), color="#5be0a0", lw=1.5, linestyle="--",
                           label=f"Role avg: ${subset.mean():,.0f}")
                ax.set_xlabel("Salary (USD)", color="#9ba8c2")
                ax.set_ylabel("Count", color="#9ba8c2")
                ax.tick_params(colors="#9ba8c2")
                for spine in ax.spines.values(): spine.set_edgecolor("#2e3a5c")
                ax.legend(fontsize=8, labelcolor="white",
                          facecolor="#1e2130", edgecolor="#2e3a5c")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda x, _: f"${x/1000:.0f}k"))
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with col_info:
        st.subheader(" Salary Growth by Experience")
        exp_range = np.arange(0, 21)
        preds_exp = [predict_salary(job_role, country, e, exp_level,
                                    work_mode, company_sz, education, industry, weekly_hrs)
                     for e in exp_range]

        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
        ax2.set_facecolor("#161a28")
        ax2.plot(exp_range, preds_exp, color="#4a7eda", lw=2.5, marker="o",
                 markersize=4, markerfacecolor="#f5a623")
        ax2.fill_between(exp_range, preds_exp, alpha=0.15, color="#4a7eda")
        ax2.axvline(exp_yrs, color="#f5a623", linestyle="--", alpha=0.8,
                    label=f"Your position ({exp_yrs} yrs)")
        ax2.set_xlabel("Years of Experience", color="#9ba8c2")
        ax2.set_ylabel("Predicted Salary (USD)", color="#9ba8c2")
        ax2.tick_params(colors="#9ba8c2")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2e3a5c")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1000:.0f}k"))
        ax2.legend(fontsize=9, labelcolor="white",
                   facecolor="#1e2130", edgecolor="#2e3a5c")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Work mode comparison
        st.subheader(" Work Mode Salary Impact")
        modes   = sorted(df["work_mode"].unique())
        mode_preds = [predict_salary(job_role, country, exp_yrs, exp_level,
                                     m, company_sz, education, industry, weekly_hrs)
                      for m in modes]
        fig3, ax3 = plt.subplots(figsize=(5, 2.5), facecolor="#0e1117")
        ax3.set_facecolor("#161a28")
        colors = ["#f5a623" if m == work_mode else "#4a7eda" for m in modes]
        ax3.barh(modes, mode_preds, color=colors, edgecolor="#2a3a6a")
        ax3.set_xlabel("Predicted Salary (USD)", color="#9ba8c2")
        ax3.tick_params(colors="#9ba8c2")
        for spine in ax3.spines.values(): spine.set_edgecolor("#2e3a5c")
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig3, use_container_width=True)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(" Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    # Salary distribution
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        ax.hist(df["salary_usd"], bins=50, color="#4a7eda", edgecolor="#2a3a6a", alpha=0.85)
        ax.set_title("Global Salary Distribution", color="white", fontsize=12)
        ax.set_xlabel("Salary (USD)", color="#9ba8c2")
        ax.set_ylabel("Count", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    # Top-paying roles
    with c2:
        role_sal = df.groupby("job_role")["salary_usd"].median().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        bars = ax.barh(role_sal.index, role_sal.values, color="#5be0a0", edgecolor="#2a3a6a", alpha=0.9)
        ax.set_title("Median Salary by Job Role", color="white", fontsize=12)
        ax.set_xlabel("Median Salary (USD)", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    c3, c4 = st.columns(2)

    # Salary by experience level boxplot
    with c3:
        order = ["Entry", "Mid", "Senior", "Lead"]
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        data_bp = [df[df["experience_level"] == lvl]["salary_usd"].values for lvl in order]
        bp = ax.boxplot(data_bp, labels=order, patch_artist=True,
                        medianprops=dict(color="#f5a623", lw=2))
        colors_bp = ["#4a7eda", "#5be0a0", "#f5a623", "#e05b5b"]
        for patch, c in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]: item.set_color("#9ba8c2")
        ax.set_title("Salary by Experience Level", color="white", fontsize=12)
        ax.set_xlabel("Level", color="#9ba8c2"); ax.set_ylabel("Salary (USD)", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    # Salary vs experience scatter
    with c4:
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        sc = ax.scatter(sample["experience_years"], sample["salary_usd"],
                        alpha=0.25, s=8, c=sample["salary_usd"],
                        cmap="cool", norm=plt.Normalize(28000, 300000))
        ax.set_title("Salary vs Experience Years", color="white", fontsize=12)
        ax.set_xlabel("Experience (years)", color="#9ba8c2")
        ax.set_ylabel("Salary (USD)", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    num_cols = ["salary_usd", "experience_years", "weekly_hours", "company_rating",
                "ai_adoption_score", "skill_demand_score", "job_security_score",
                "career_growth_score", "cost_of_living_index", "economic_index", "bonus_usd"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0e1117")
    ax.set_facecolor("#161a28")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, linecolor="#2e3a5c", ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="#9ba8c2", labelsize=8)
    ax.set_title("Correlation Heatmap", color="white", pad=10, fontsize=12)
    st.pyplot(fig, use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(" Model Performance")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score",  f"{model_metrics['r2']:.4f}")
    mc2.metric("Mean Absolute Error", f"${model_metrics['mae']:,.0f}")
    mc3.metric("Training Records", f"{len(df)*0.8:,.0f}")

    col_a, col_b = st.columns(2)

    with col_a:
        # Actual vs predicted
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        ax.scatter(y_test, y_pred, alpha=0.2, s=6, color="#4a7eda")
        lims = [max(y_test.min(), y_pred.min()), min(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, color="#f5a623", lw=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Salary", color="#9ba8c2")
        ax.set_ylabel("Predicted Salary", color="#9ba8c2")
        ax.set_title("Actual vs Predicted", color="white", fontsize=12)
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.legend(fontsize=9, labelcolor="white", facecolor="#1e2130", edgecolor="#2e3a5c")
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        # Feature importance
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top10 = importance.sort_values(ascending=True).tail(10)
        labels = [c.replace("_enc", "").replace("_", " ").title() for c in top10.index]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        ax.barh(labels, top10.values, color="#5be0a0", edgecolor="#2a3a6a", alpha=0.9)
        ax.set_title("Top 10 Feature Importances", color="white", fontsize=12)
        ax.set_xlabel("Importance Score", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        st.pyplot(fig, use_container_width=True); plt.close()

    # Residuals
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0e1117")
    ax.set_facecolor("#161a28")
    ax.hist(residuals, bins=60, color="#e05b5b", edgecolor="#2a3a6a", alpha=0.8)
    ax.axvline(0, color="#f5a623", lw=2)
    ax.set_title("Residual Distribution (Actual − Predicted)", color="white", fontsize=12)
    ax.set_xlabel("Residual (USD)", color="#9ba8c2")
    ax.tick_params(colors="#9ba8c2")
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    st.pyplot(fig, use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNTRY COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🗺️ Country Salary Comparison")

    country_stats = df.groupby("country").agg(
        median_salary=("salary_usd", "median"),
        mean_salary=("salary_usd", "mean"),
        job_count=("salary_usd", "count"),
        avg_experience=("experience_years", "mean"),
    ).reset_index().sort_values("median_salary", ascending=False)

    # Table
    st.dataframe(
        country_stats.style
            .format({"median_salary": "${:,.0f}", "mean_salary": "${:,.0f}",
                     "job_count": "{:,}", "avg_experience": "{:.1f} yrs"})
            .background_gradient(subset=["median_salary"], cmap="Blues"),
        use_container_width=True, hide_index=True
    )

    c_a, c_b = st.columns(2)

    with c_a:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        ax.barh(country_stats["country"], country_stats["median_salary"],
                color="#4a7eda", edgecolor="#2a3a6a", alpha=0.9)
        ax.set_title("Median Salary by Country", color="white", fontsize=12)
        ax.set_xlabel("Median Salary (USD)", color="#9ba8c2")
        ax.tick_params(colors="#9ba8c2")
        for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig, use_container_width=True); plt.close()

    with c_b:
        # Role × Country heatmap (mean salary)
        pivot = df.pivot_table(values="salary_usd", index="job_role",
                               columns="country", aggfunc="median")
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
        ax.set_facecolor("#161a28")
        sns.heatmap(pivot / 1000, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="#2e3a5c", ax=ax,
                    annot_kws={"size": 7})
        ax.set_title("Median Salary (k USD): Role × Country", color="white", fontsize=11)
        ax.tick_params(colors="#9ba8c2", labelsize=7)
        ax.set_xlabel("", color="#9ba8c2")
        ax.set_ylabel("", color="#9ba8c2")
        st.pyplot(fig, use_container_width=True); plt.close()

    # Predict across all countries for current role
    st.subheader(f"🔮 Your Role ({job_role}) Predicted Salary Across All Countries")
    all_countries = sorted(df["country"].unique())
    country_preds = [(c, predict_salary(job_role, c, exp_yrs, exp_level,
                                        work_mode, company_sz, education, industry, weekly_hrs))
                     for c in all_countries]
    cp_df = pd.DataFrame(country_preds, columns=["Country", "Predicted Salary"]).sort_values(
        "Predicted Salary", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#161a28")
    bar_colors = ["#f5a623" if c == country else "#4a7eda" for c in cp_df["Country"]]
    ax.barh(cp_df["Country"], cp_df["Predicted Salary"], color=bar_colors, edgecolor="#2a3a6a")
    ax.set_title(f"Predicted Salary: {job_role} ({exp_yrs} yrs exp)", color="white", fontsize=12)
    ax.set_xlabel("Predicted Salary (USD)", color="#9ba8c2")
    ax.tick_params(colors="#9ba8c2")
    for sp in ax.spines.values(): sp.set_edgecolor("#2e3a5c")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    st.pyplot(fig, use_container_width=True); plt.close()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.85rem;'>"
    " AI Jobs Salary Dashboard · Powered by Random Forest · 90,000+ records"
    "</p>",
    unsafe_allow_html=True
)
