"""
🔍 Pipeline Leak Detection Dashboard with Groq LLM Chat
Complete end-to-end Streamlit app. Deploy to GitHub Codespaces/Streamlit Cloud.
Requirements: pip install streamlit groq pandas matplotlib seaborn openpyxl
NO API KEY IN CODE - Enter in UI for security.
Files needed: df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq

# Page config
st.set_page_config(
    page_title="Leak Detection Dashboard + LLM Chat", 
    layout="wide",
    page_icon="🔍"
)

plt.rcParams['figure.max_open_warning'] = 50
st.title("🔍 Pipeline Leak Detection: Analysis + Groq LLM Chat")

# --- GROQ API KEY INPUT (Secure UI entry) ---
with st.sidebar:
    st.header("🔑 API Configuration")
    groq_api_key = st.text_input(
        "Enter your Groq API Key (gsk_...):",
        type="password",
        help="Get free key from https://console.groq.com/keys | Starts with 'gsk_'"
    )
    
    if st.button("✅ Test Groq Connection", type="primary"):
        if not groq_api_key or not groq_api_key.startswith("gsk_"):
            st.error("❌ Invalid key format. Must start with 'gsk_'")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "Say 'Connected!'"}],
                    max_tokens=10
                )
                st.success(f"✅ Groq Connected! Model: {response.model}")
                st.session_state.groq_client = client
                st.session_state.api_key_valid = True
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)[:100]}...")

# --- DATA UPLOAD & PREPROCESSING ---
uploaded_files = st.file_uploader(
    "📁 Upload Excel files (df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx):",
    type=['xlsx'], accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Please upload your Excel files to start analysis")
    st.stop()

# Load and validate files
file_dict = {f.name: f for f in uploaded_files}
required_files = ['df_pidws.xlsx', 'df_manual_digging.xlsx', 'df_lds_IV.xlsx']

missing_files = [f for f in required_files if f not in file_dict]
if missing_files:
    st.error(f"❌ Missing files: {', '.join(missing_files)}")
    st.stop()

@st.cache_data
def load_and_preprocess(file_dict):
    """Safe data loading with preprocessing"""
    df_pidws = pd.read_excel(file_dict['df_pidws.xlsx'])
    df_manual_digging = pd.read_excel(file_dict['df_manual_digging.xlsx']) 
    df_lds_IV = pd.read_excel(file_dict['df_lds_IV.xlsx'])
    
    # Preprocessing
    df_manual_digging['DateTime'] = pd.to_datetime(df_manual_digging['DateTime'])
    df_manual_digging['Date_new'] = df_manual_digging['DateTime'].dt.date
    df_manual_digging['Time_new'] = df_manual_digging['DateTime'].dt.time
    
    df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
    df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
    df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
    
    return df_pidws, df_manual_digging, df_lds_IV

try:
    df_pidws, df_manual_digging, df_lds_IV = load_and_preprocess(file_dict)
    
    # Store in session state
    st.session_state.dfs = {
        'manual_digging': df_manual_digging,
        'lds_iv': df_lds_IV,
        'pidws': df_pidws
    }
    
    st.session_state.df_summaries = {
        name: df.describe(include='all').to_dict() for name, df in st.session_state.dfs.items()
    }
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("⛏️ Digging Events", f"{len(df_manual_digging):,}")
    col2.metric("💧 Leak Events", f"{len(df_lds_IV):,}")
    col3.metric("📊 PIDWS Records", f"{len(df_pidws):,}")
    
    st.success("✅ Data loaded and preprocessed!")
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Interactive Analysis", "🔗 All Chainages", "💬 LLM Data Chat"])

# Tab 1: Interactive Chainage Analysis
with tab1:
    st.header("🎯 Chainage-Specific Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        target_chainage = st.number_input("Target Chainage (km):", value=25.4, min_value=0.0, step=0.1)
    with col2:
        tolerance = st.number_input("Tolerance (±km):", value=1.0, min_value=0.1, step=0.1)
    
    if st.button("🚀 Analyze Chainage", type="primary"):
        df_digging_filtered = df_manual_digging[
            abs(df_manual_digging['Original_chainage'] - target_chainage) <= tolerance
        ].copy()
        
        df_leaks_filtered = df_lds_IV[
            abs(df_lds_IV['chainage'] - target_chainage) <= tolerance
        ].copy()
        
        col1, col2 = st.columns(2)
        col1.metric("🔵 Digging Events", len(df_digging_filtered))
        col2.metric("🔴 Leak Events", len(df_leaks_filtered))
        
        # Scatter plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        if not df_digging_filtered.empty:
            sns.scatterplot(
                data=df_digging_filtered, x='DateTime', y='Original_chainage',
                color='blue', label='Digging', marker='o', s=80, ax=ax
            )
        
        if not df_leaks_filtered.empty:
            sns.scatterplot(
                data=df_leaks_filtered, x='DateTime', y='chainage',
                color='red', label='Leaks', marker='X', s=120, ax=ax
            )
        
        plt.title(f'Digging vs Leaks: Chainage {target_chainage} ±{tolerance}km', fontsize=16, pad=20)
        plt.xlabel('DateTime', fontsize=12)
        plt.ylabel('Chainage (km)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Timeline table
        if not df_digging_filtered.empty or not df_leaks_filtered.empty:
            st.subheader("📋 Event Timeline")
            timeline = pd.concat([
                df_digging_filtered[['DateTime', 'Original_chainage']].assign(Type='Digging'),
                df_leaks_filtered[['DateTime', 'chainage']].rename(columns={'chainage': 'Original_chainage'}).assign(Type='Leak')
            ]).sort_values('DateTime').reset_index(drop=True)
            st.dataframe(timeline.head(20))

# Tab 2: All Chainages Analysis  
with tab2:
    st.header("🌐 Complete Chainage Analysis")
    
    unique_chainages = sorted(df_manual_digging['Original_chainage'].unique())
    st.info(f"Found {len(unique_chainages)} unique chainages")
    
    tolerance_all = st.slider("Tolerance (km)", 0.1, 2.0, 1.0)
    
    # Summary table
    chainage_analysis = []
    for chainage in unique_chainages[:50]:  # Limit for performance
        digs = len(df_manual_digging[abs(df_manual_digging['Original_chainage'] - chainage) <= tolerance_all])
        leaks = len(df_lds_IV[abs(df_lds_IV['chainage'] - chainage) <= tolerance_all])
        chainage_analysis.append({
            'Chainage': f"{chainage:.1f}", 
            'Digging': digs, 
            'Leaks': leaks, 
            'Match Score': digs * leaks
        })
    
    df_analysis = pd.DataFrame(chainage_analysis).sort_values('Match Score', ascending=False)
    st.dataframe(df_analysis.head(20))
    
    # Top matches chart
    top_matches = df_analysis.head(10)
    st.subheader("🏆 Top Chainages by Correlation")
    st.bar_chart(top_matches.set_index('Chainage')[['Match Score']])

# Tab 3: LLM Chat (only if API connected)
with tab3:
    st.header("💬 AI Pipeline Analyst (Groq Llama 3.3)")
    
    if not groq_api_key or 'api_key_valid' not in st.session_state:
        st.warning("⚠️ Enter & test Groq API key in sidebar first")
        st.stop()
    
    client = st.session_state.groq_client
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Data context
    data_context = f"""
    Indian Oil Pipeline Leak Detection Analysis:
    - Manual Digging: {len(df_manual_digging)} events (Original_chainage km, DateTime, Event Duration)
    - LDS-IV Leaks: {len(df_lds_IV)} detections (chainage km, DateTime)
    - PIDWS: {len(df_pidws)} sensor records
    
    Mission: Correlate digging events with actual leaks by chainage & time.
    Provide: Insights, correlations, risk chainages, matplotlib plot code, anomaly detection ideas.
    Example queries: "Top 5 risky chainages", "Temporal correlation", "Plot code for chainage 25"
    """
    
    # Chat input
    if prompt := st.chat_input("💭 Ask about leak correlations, risky chainages, or plot code..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        messages = [{"role": "system", "content": data_context}]
        for msg in st.session_state.messages:
            messages.append(msg)
        
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                stream=True,
            )
            response = st.write_stream(
                chunk.choices[0].delta.content or "" for chunk in stream
            )
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("⚡ Powered by", "Groq LPU")
col2.metric("📈 Framework", "Streamlit")
col3.metric("🔬 Domain", "Pipeline Integrity")

st.markdown("""
*For Indian Oil Corporation pipeline operations - Leak detection & correlation analysis*
""")
