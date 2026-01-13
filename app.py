import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Streamlit page config
st.set_page_config(
    page_title="Pipeline Pilferage Classification",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Pilferage Detection & Classification")
st.markdown("**Interactive analysis of PIDWS digging events vs LDS leak detection**")

# Sidebar for file uploads and parameters
st.sidebar.header("📁 Data Upload")
uploaded_pidws = st.sidebar.file_uploader("Upload PIDWS data (df_pidws_III.xlsx)", type=['xlsx', 'xls'])
uploaded_lds = st.sidebar.file_uploader("Upload LDS data (df_lds_III.xlsx)", type=['xlsx', 'xls'])

st.sidebar.header("⚙️ Classification Parameters")
chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
time_window = st.sidebar.slider("Time Window (hours after digging)", 1, 72, 48, 1)

if st.sidebar.button("🔄 Run Classification"):
    if uploaded_pidws is not None and uploaded_lds is not None:
        with st.spinner("Processing data and running classification..."):
            # Load data from uploaded files
            df_pidws = pd.read_excel(uploaded_pidws)
            df_lds = pd.read_excel(uploaded_lds)
            
            # Parse PIDWS datetime and duration
            df_pidws['DateTime'] = pd.to_datetime(df_pidws['Date'] + ' ' + df_pidws['Time'], format='%d-%m-%Y %H:%M:%S')

            @st.cache_data
            def parse_duration(dur_str):
                if pd.isna(dur_str):
                    return pd.Timedelta(0)
                dur_str = str(dur_str).strip().lower().replace(' ', '')
                mins, secs = 0, 0
                if 'm' in dur_str:
                    m_part = dur_str.split('m')[0]
                    if m_part.isdigit():
                        mins = int(m_part)
                    dur_str = dur_str.split('m')[1]
                if 's' in dur_str:
                    s_part = dur_str.replace('s', '')
                    if s_part.isdigit():
                        secs = int(s_part)
                return pd.Timedelta(minutes=mins, seconds=secs)

            df_pidws['duration_td'] = df_pidws['Event Duration'].apply(parse_duration)
            df_pidws['end_time'] = df_pidws['DateTime'] + df_pidws['duration_td']

            # Parse LDS datetime
            df_lds['DateTime'] = pd.to_datetime(df_lds['Date'].astype(str) + ' ' + df_lds['Time'])

            # Classification function
            @st.cache_data
            def classify_pilferage(pidws_df, lds_df, chainage_tol, time_window_hours):
                classified = []
                for _, event in pidws_df.iterrows():
                    window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
                    mask = (lds_df['DateTime'] > window_end) & \
                           (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
                    matches = lds_df[mask].copy()
                    if not matches.empty:
                        matches['linked_event_time'] = event['DateTime']
                        matches['linked_chainage'] = event['chainage']
                        matches['pilferage_score'] = 1 / (1 + (matches['DateTime'] - window_end).dt.total_seconds() / 3600)
                        classified.append(matches)
                return pd.concat(classified, ignore_index=True) if classified else pd.DataFrame()

            # Run classification
            pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window)
            
            # Classify LDS dataset
            df_lds_classified = df_lds.copy()
            df_lds_classified['is_pilferage'] = False
            if not pilferage_leaks.empty:
                pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
                mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
                    pilferage_ids.set_index(['DateTime', 'chainage']).index
                )
                df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
            
            # Store in session state
            st.session_state.df_pidws = df_pidws
            st.session_state.df_lds = df_lds
            st.session_state.pilferage_leaks = pilferage_leaks
            st.session_state.df_lds_classified = df_lds_classified
            st.session_state.classification_run = True

# Display results only if data is processed
if 'classification_run' in st.session_state and st.session_state.classification_run:
    df_pidws = st.session_state.df_pidws
    df_lds = st.session_state.df_lds
    pilferage_leaks = st.session_state.pilferage_leaks
    df_lds_classified = st.session_state.df_lds_classified
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total LDS Leaks", len(df_lds))
        st.metric("Detected Pilferage", len(pilferage_leaks), f"{100*len(pilferage_leaks)/len(df_lds):.1f}%")
        st.metric("PIDWS Events", len(df_pidws))
    
    with col2:
        if not pilferage_leaks.empty:
            avg_score = pilferage_leaks['pilferage_score'].mean()
            st.metric("Avg Pilferage Score", f"{avg_score:.3f}")
            st.metric("Max Leak Size (Pilferage)", f"{pilferage_leaks['leak size'].max():.1f}")
    
    # Summary statistics
    st.subheader("📊 Classification Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Leak Classification**")
        class_summary = df_lds_classified['is_pilferage'].value_counts().to_frame()
        st.dataframe(class_summary.T, use_container_width=True)
    
    with col2:
        if not pilferage_leaks.empty:
            st.markdown("**Top Chainage Clusters**")
            chainage_stats = pilferage_leaks.groupby('linked_chainage')['leak size'].agg(['count', 'mean', 'max']).round(1)
            st.dataframe(chainage_stats.head(), use_container_width=True)
    
    with col3:
        st.markdown("**Pilferage Stats**")
        if not pilferage_leaks.empty:
            st.dataframe(pilferage_leaks[['leak size', 'pilferage_score']].describe().round(2), use_container_width=True)
    
    # Interactive visualizations
    st.subheader("📈 Interactive Visualizations")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chainage Distribution', 'Temporal Patterns', 'Leak Size Distribution', 'Spatiotemporal View'),
        specs=[[{"type": "histogram"}, {"secondary_y": False}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # 1. Chainage distribution
    fig.add_trace(
        go.Histogram(x=df_pidws['chainage'], name="PIDWS Digging", opacity=0.7, marker_color="#FF9500"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df_lds['chainage'], name="All Leaks", opacity=0.7, marker_color="#007AFF"),
        row=1, col=1
    )
    if not pilferage_leaks.empty:
        fig.add_vline(x=pilferage_leaks['linked_chainage'].mean(), line_dash="dash", 
                      line_color="red", annotation_text="Pilferage Mean", row=1, col=1)
    
    # 2. Temporal patterns
    all_events = pd.concat([
        df_pidws[['DateTime', 'chainage']].assign(type='Digging'),
        df_lds[['DateTime', 'chainage']].assign(type='Leak'),
        pilferage_leaks[['DateTime', 'linked_chainage']].rename(columns={'linked_chainage':'chainage'}).assign(type='Pilferage')
    ], ignore_index=True)
    
    time_counts = all_events.groupby([all_events['DateTime'].dt.floor('H'), 'type']).size().unstack(fill_value=0)
    for event_type in time_counts.columns:
        fig.add_trace(
            go.Scatter(x=time_counts.index, y=time_counts[event_type], 
                      name=event_type, mode='lines+markers'),
            row=1, col=2
        )
    
    # 3. Leak size boxplot
    fig.add_trace(
        go.Box(y=df_lds_classified[df_lds_classified['is_pilferage'] == False]['leak size'], 
               name="Other Leaks", marker_color="#007AFF"), row=2, col=1
    )
    if not pilferage_leaks.empty:
        fig.add_trace(
            go.Box(y=pilferage_leaks['leak size'], name="Pilferage", marker_color="#FF3B30"), row=2, col=1
        )
    
    # 4. Spatiotemporal scatter
    colors = ['red' if x else 'blue' for x in df_lds_classified['is_pilferage']]
    fig.add_trace(
        go.Scatter(x=df_lds_classified['DateTime'], y=df_lds_classified['chainage'],
                  mode='markers', marker=dict(color=colors, size=6, opacity=0.6),
                  name="Leaks (Red=Pilferage)"), row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Pipeline Event Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Download buttons
    st.subheader("💾 Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_buffer = io.StringIO()
        df_lds_classified.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Classified LDS",
            csv_buffer.getvalue(),
            "lds_classified.csv",
            "text/csv"
        )
    
    with col2:
        if not pilferage_leaks.empty:
            csv_buffer = io.StringIO()
            pilferage_leaks.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Pilferage Leaks",
                csv_buffer.getvalue(),
                "pilferage_leaks.csv",
                "text/csv"
            )
    
    with col3:
        st.download_button(
            "Download Summary Report",
            "Classification complete! Check CSV files above.",
            "summary.txt",
            "text/plain"
        )

else:
    st.info("👆 Please upload your PIDWS and LDS Excel files and click 'Run Classification'")
    st.markdown("""
    ### 📋 Expected File Structure
    **df_pidws_III.xlsx columns:**
    - Date, Time, chainage, Event Duration
    
    **df_lds_III.xlsx columns:**
    - Date, Time, chainage, leak size
    """)
