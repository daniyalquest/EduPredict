import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import bcrypt
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

# ===============================
# 1. CONFIGURATION & DATABASE
# ===============================
st.set_page_config(
    page_title="EduPredict Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_db():
    uri = "mongodb+srv://daniyal2472_db_user:Nc1mhy0mORO9RTLY@cluster-edupredict.wggn6ki.mongodb.net/?appName=Cluster-EduPredict"
    client = MongoClient(uri)
    return client.edu_predict_db

db = init_db()
users_col = db.users
records_col = db.student_records

# ===============================
# 2. AUTHENTICATION HELPERS
# ===============================
def hash_pw(pw):
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())

def check_pw(pw, hashed):
    return bcrypt.checkpw(pw.encode("utf-8"), hashed)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "show_login_toast" not in st.session_state:
    st.session_state.show_login_toast = False

# ===============================
# 3. LOAD ML ASSETS
# ===============================
@st.cache_resource
def load_ml():
    if os.path.exists("student_model.pkl") and os.path.exists("encoders.pkl"):
        return joblib.load("student_model.pkl"), joblib.load("encoders.pkl")
    return None, None

model, encoders = load_ml()

# ===============================
# 4. ANALYTICS & EDA
# ===============================
def show_analytics():
    st.title("📊 Analytics & EDA Dashboard")

    with st.spinner("Fetching data from MongoDB Atlas..."):
        raw_data = list(records_col.find({}, {"_id": 0}))
        if not raw_data:
            st.warning("No data found in database.")
            return
        df = pd.DataFrame(raw_data)

    # ---- KPIs ----
    st.subheader("📈 Subject Performance Averages")
    subjects = ["math_score", "science_score", "english_score", "overall_score"]
    cols = st.columns(len(subjects))
    for i, sub in enumerate(subjects):
        cols[i].metric(sub.replace("_", " ").title(), f"{df[sub].mean():.2f}%")

    st.divider()
    st.subheader("🔍 Deep Data Insights")

    # Layout Columns
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r3c1, r3c2 = st.columns(2)
    r4c1, r4c2 = st.columns(2)
    r5c1, r5c2 = st.columns(2)

    with r1c1:
        st.write("**1. Overall Score Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df["overall_score"], kde=True, color="#6366f1", ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Shows score frequency. A bell curve indicates balanced performance across the student body.")

    with r1c2:
        st.write("**2. Study Hours vs Overall Score**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="study_hours", y="overall_score", color="#6366f1", alpha=0.4, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Identifies the correlation between time invested and academic results.")

    with r2c1:
        st.write("**3. Attendance Impact**")
        fig, ax = plt.subplots()
        sns.regplot(data=df, x="attendance_percentage", y="overall_score", 
                    scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** A strong upward trend confirms that consistent attendance is critical for success.")

    with r2c2:
        st.write("**4. Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.select_dtypes(np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Highlights which numerical factors (like Math vs Science) are most strongly linked.")

    with r3c1:
        st.write("**5. Gender-wise Performance**")
        fig, ax = plt.subplots()
        sns.boxplot(x="gender", y="overall_score", data=df, palette="muted", ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Compares the median and spread of scores across genders.")

    with r3c2:
        st.write("**6. Parent Education Impact**")
        fig, ax = plt.subplots()
        sns.barplot(x="parent_education", y="overall_score", data=df, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.info("💡 **Insight:** Visualizes how the educational background of parents correlates with achievement.")

    with r4c1:
        st.write("**7. Final Grade by School Type**")
        fig, ax = plt.subplots()
        sns.countplot(x="final_grade", hue="school_type", data=df, 
                      order=sorted(df["final_grade"].unique()), ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Public vs Private grade distribution comparison.")

    with r4c2:
        st.write("**8. Study Method Effectiveness**")
        fig, ax = plt.subplots()
        sns.violinplot(x="study_method", y="overall_score", data=df, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Wider sections indicate the most common score ranges for each method.")

    with r5c1:
        st.write("**9. Subject Score Comparison**")
        fig, ax = plt.subplots()
        melted = df.melt(value_vars=["math_score", "science_score", "english_score"],
                         var_name="Subject", value_name="Score")
        sns.boxplot(x="Subject", y="Score", data=melted, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Identifies which subjects students find most difficult overall.")

    with r5c2:
        st.write("**10. Travel Time vs Performance**")
        fig, ax = plt.subplots()
        sns.pointplot(x="travel_time", y="overall_score", data=df, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Analyzes if long commute times negatively impact energy and results.")

# ===============================
# 5. PREDICTION
# ===============================
def show_prediction():
    st.title("🔮 AI Score Predictor")

    if not model:
        st.error("Model not found.")
        return

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 10, 30, 17)
            gender = st.selectbox("Gender", encoders["gender"].classes_)
            school = st.selectbox("School Type", encoders["school_type"].classes_)
            parent = st.selectbox("Parent Education", encoders["parent_education"].classes_)
        with c2:
            hours = st.slider("Study Hours", 0.0, 20.0, 8.0)
            attn = st.slider("Attendance %", 0.0, 100.0, 80.0)
            net = st.selectbox("Internet Access", encoders["internet_access"].classes_)
            travel = st.selectbox("Travel Time", encoders["travel_time"].classes_)
        with c3:
            ext = st.selectbox("Extra Activities", encoders["extra_activities"].classes_)
            meth = st.selectbox("Study Method", encoders["study_method"].classes_)
            m = st.slider("Math Score", 0, 100, 60)
            s = st.slider("Science Score", 0, 100, 60)
            e = st.slider("English Score", 0, 100, 60)

        if st.form_submit_button("Predict Performance", use_container_width=True):
            df_input = pd.DataFrame([{
                "age": age, "gender": gender, "school_type": school,
                "parent_education": parent, "study_hours": hours,
                "attendance_percentage": attn, "internet_access": net,
                "travel_time": travel, "extra_activities": ext,
                "study_method": meth, "math_score": m, "science_score": s, "english_score": e
            }])

            for col, le in encoders.items():
                df_input[col] = le.transform(df_input[col])

            prediction = model.predict(df_input)[0]
            st.metric("Predicted Overall Score", f"{prediction:.2f}%")
            st.toast("✅ Prediction generated!")

# ===============================
# 6. DATABASE EXPLORER
# ===============================
def show_db_explorer():
    st.title("📂 Database Explorer")

    # Fetch data and exclude the MongoDB '_id' field
    raw_data = list(records_col.find({}, {"_id": 0}))
    
    if not raw_data:
        st.warning("Database is empty.")
        return

    df = pd.DataFrame(raw_data)

    # ---- Summary Metrics (Mean of all numerical columns) ----
    st.subheader("📊 Numerical Column Averages")
    
    # Select only numeric columns (int and float)
    num_df = df.select_dtypes(include=[np.number])
    
    if not num_df.empty:
        # Create a grid to display metrics (max 4 per row for readability)
        num_cols = num_df.columns
        rows = [num_cols[i:i + 4] for i in range(0, len(num_cols), 4)]
        
        for row in rows:
            cols = st.columns(len(row))
            for i, col_name in enumerate(row):
                avg_val = num_df[col_name].mean()
                display_name = col_name.replace("_", " ").title()
                cols[i].metric(display_name, f"{avg_val:.2f}")
    
    st.divider()

    # ---- Full Data Table ----
    st.subheader(f"📋 Full Records ({len(df)} entries)")
    
    # Search Bar for table filtering
    search = st.text_input("🔍 Search records by any value", "")
    if search:
        # Filter dataframe based on search string
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        df_display = df[mask]
    else:
        df_display = df

    st.dataframe(df_display, use_container_width=True, height=500)

    # Download Option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Data as CSV",
        data=csv,
        file_name='edupredict_records.csv',
        mime='text/csv',
    )

# ===============================
# 7. MAIN APP ROUTING
# ===============================
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")

        if st.button("Login", use_container_width=True):
            if not u.strip() or not p.strip():
                st.toast("ℹ️ Please enter both username and password.")
            else:
                user = users_col.find_one({"username": u.strip()})
                if user and check_pw(p, user["password"]):
                    st.session_state.logged_in = True
                    st.session_state.username = u.strip()
                    st.session_state.show_login_toast = True 
                    st.rerun()
                else:
                    st.toast("⚠️ Invalid username or password.")

    with tab2:
        nu = st.text_input("New Username", key="reg_u")
        npw = st.text_input("New Password", type="password", key="reg_p")
        cpw = st.text_input("Confirm Password", type="password", key="reg_cp")

        if st.button("Sign Up", use_container_width=True):
            if not nu.strip() or not npw.strip():
                st.toast("⚠️ Fields cannot be empty.")
            elif npw != cpw:
                st.toast("⚠️ Passwords do not match.")
            elif len(npw) < 6:
                st.toast("⚠️ Password too short.")
            elif users_col.find_one({"username": nu.strip()}):
                st.toast(f"⚠️ Username '{nu}' taken.")
            else:
                users_col.insert_one({"username": nu.strip(), "password": hash_pw(npw)})
                st.toast("✅ Account created!")
                st.balloons()

else:
    if st.session_state.show_login_toast:
        st.toast(f"✅ Welcome back, {st.session_state.username}!")
        st.session_state.show_login_toast = False 

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        page = st.radio("Navigation", ["Dashboard", "Predictor", "Database Explorer"])
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    if page == "Dashboard":
        show_analytics()
    elif page == "Predictor":
        show_prediction()
    else:
        show_db_explorer()