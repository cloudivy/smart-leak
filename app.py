%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from io import BytesIO
from groq import Groq

# Configure matplotlib to prevent memory warnings
plt.rcParams['figure.max_open_warning'] = 50

# Set Streamlit page configuration
st.set_page_config(page_title="Leak Detection Dashboard", layout="wide")
st.title("🔍 Identifying and implementing newer methods of quick detection of location of leaks in the mainline ")

# Initialize session state for dataframes and analysis results
if 'df_pidws' not in st.session_state:
    st.session_state['df_pidws'] = pd.DataFrame()
    st.session_state['df_manual_digging'] = pd.DataFrame()
    st.session_state['df_lds_IV'] = pd.DataFrame()
    st.session_state['analysis_summary'] = ""
    st.session_state['target_chainage'] = 25.4
    st.session_state['tolerance'] = 1.0
    st.session_state['run_all_chainage_analysis_flag'] = False # For the checkbox
    st.session_state['tolerance_all'] = 1.0
    st.session_state['unique_chainages'] = []

# --- Enhanced data loading (SAFE - works without repo files) ---
tab1, tab2, tab3 = st.tabs(["📁 Upload Files", "📊 Analysis", "💬 LLM Chat"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload Excel files:",
        type=['xlsx'], accept_multiple_files=True,
        help="Upload: df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx"
    )

    if uploaded_files:
        file_dict = {f.name: f for f in uploaded_files}
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

        # Load dataframes safely
        try:
            st.session_state['df_pidws'] = pd.read_excel(file_dict['df_pidws.xlsx']) if 'df_pidws.xlsx' in file_dict else pd.DataFrame()
            st.session_state['df_manual_digging'] = pd.read_excel(file_dict['df_manual_digging.xlsx']) if 'df_manual_digging.xlsx' in file_dict else pd.DataFrame()
            st.session_state['df_lds_IV'] = pd.read_excel(file_dict['df_lds_IV.xlsx']) if 'df_lds_IV.xlsx' in file_dict else pd.DataFrame()

            col1, col2, col3 = st.columns(3)
            col1.metric("Digging Data", f"{len(st.session_state['df_manual_digging']):,}", delta="✅")
            col2.metric("Leak Data", f"{len(st.session_state['df_lds_IV']):,}", delta="✅")
            col3.metric("PIDWS Data", f"{len(st.session_state['df_pidws']):,}", delta="✅")

        except Exception as e:
            st.error(f"❌ Error reading files: {e}")
    else:
        st.info("👆 Upload your Excel files to start analysis")

# Retrieve dataframes from session state for use in other tabs
df_pidws = st.session_state['df_pidws']
df_manual_digging = st.session_state['df_manual_digging']
df_lds_IV = st.session_state['df_lds_IV']

# --- Analysis Tab ---
with tab2:
    st.write("Imported matplotlib.pyplot as plt and seaborn as sns.")

    # Data previews
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Manual Digging Data")
        if not df_manual_digging.empty and all(col in df_manual_digging.columns for col in ['Event Duration', 'Original_chainage', 'DateTime']):
            st.dataframe(df_manual_digging[['Event Duration', 'Original_chainage', 'DateTime']].head())
        elif not df_manual_digging.empty:
            st.dataframe(df_manual_digging.head())
        else:
            st.info("No manual digging data available. Please upload in 'Upload Files' tab.")

    with col2:
        st.subheader("LDS IV Data")
        if not df_lds_IV.empty:
            st.dataframe(df_lds_IV.head())
        else:
            st.info("No LDS IV data available. Please upload in 'Upload Files' tab.")

    # Data prep
    # Ensure DateTime column exists before proceeding
    if not df_manual_digging.empty and 'DateTime' in df_manual_digging.columns:
        df_manual_digging['Date_new'] = df_manual_digging['DateTime'].dt.date
        df_manual_digging['Time_new'] = df_manual_digging['DateTime'].dt.time
        st.session_state['df_manual_digging'] = df_manual_digging # Update session state
    else:
        st.warning("Warning: 'DateTime' column not found or df_manual_digging is empty. Skipping date/time parsing for df_manual_digging.")

    # LDS IV prep
    if not df_lds_IV.empty and 'Date' in df_lds_IV.columns and 'Time' in df_lds_IV.columns:
        df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
        df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
        df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
        st.session_state['df_lds_IV'] = df_lds_IV # Update session state
    else:
        st.warning("Warning: 'Date' or 'Time' column not found or df_lds_IV is empty. Skipping date/time parsing for df_lds_IV.")

    if not df_manual_digging.empty or not df_lds_IV.empty:
        st.success("✅ Data preprocessing complete")
    else:
        st.info("Data preprocessing skipped as no data is loaded.")


    # Interactive inputs
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['target_chainage'] = st.number_input(
            "🎯 Target Chainage (km):",
            value=st.session_state['target_chainage'], min_value=0.0, step=0.1
        )
    with col2:
        st.session_state['tolerance'] = st.number_input(
            "Tolerance (±km):", value=st.session_state['tolerance'], min_value=0.1, step=0.1
        )

    if st.button("🚀 Analyze Chainage", type="primary"):
        if df_manual_digging.empty and df_lds_IV.empty:
            st.error("No data loaded for analysis. Please upload files in the 'Upload Files' tab.")
        else:
            # Filter data
            df_digging_filtered = pd.DataFrame()
            if not df_manual_digging.empty and 'Original_chainage' in df_manual_digging.columns:
                df_digging_filtered = df_manual_digging[
                    abs(df_manual_digging['Original_chainage'] - st.session_state['target_chainage']) <= st.session_state['tolerance']
                ].copy()

            df_leaks_filtered = pd.DataFrame()
            if not df_lds_IV.empty and 'chainage' in df_lds_IV.columns:
                df_leaks_filtered = df_lds_IV[
                    abs(df_lds_IV['chainage'] - st.session_state['target_chainage']) <= st.session_state['tolerance']
                ].copy()

            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("🔵 Digging Events", len(df_digging_filtered))
            col2.metric("🔴 Leak Events", len(df_leaks_filtered))

            # Main scatter plot
            plt.close('all')  # Clear previous figures
            fig, ax = plt.subplots(figsize=(16, 10))

            plot_made = False
            if not df_digging_filtered.empty and 'DateTime' in df_digging_filtered.columns and 'Original_chainage' in df_digging_filtered.columns:
                sns.scatterplot(
                    data=df_digging_filtered, x='DateTime', y='Original_chainage',
                    color='blue', label='Digging Events', marker='o', s=60, ax=ax
                )
                plot_made = True

            if not df_leaks_filtered.empty and 'DateTime' in df_leaks_filtered.columns and 'chainage' in df_leaks_filtered.columns:
                sns.scatterplot(
                    data=df_leaks_filtered, x='DateTime', y='chainage',
                    color='red', label='Leak Events', marker='X', s=100, ax=ax
                )
                plot_made = True

            if plot_made:
                plt.title(f'Digging vs Leak Events - Chainage {st.session_state["target_chainage"]} ±{st.session_state["tolerance"]}km', fontsize=16, pad=20)
                plt.xlabel('DateTime', fontsize=12)
                plt.ylabel('Chainage (km)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
                st.balloons()
            else:
                st.warning("No data available with required columns to generate the main scatter plot for the selected chainage and tolerance.")

    # --- All chainages analysis (optional, collapsible) ---
    st.session_state['run_all_chainage_analysis_flag'] = st.checkbox("🔄 Show ALL Unique Chainages Analysis")

    if st.session_state['run_all_chainage_analysis_flag']:
        st.subheader("📈 Complete Chainage Analysis")

        unique_chainages_local = []
        if not df_manual_digging.empty and 'Original_chainage' in df_manual_digging.columns:
            unique_chainages_local = sorted(df_manual_digging['Original_chainage'].unique())
            st.session_state['unique_chainages'] = unique_chainages_local # Store in session state
            st.write(f"Found {len(unique_chainages_local)} unique chainages")
        else:
            st.info("No 'Original_chainage' data available for unique chainages analysis.")

        st.session_state['tolerance_all'] = st.slider("Analysis Tolerance for All Chainages", 0.1, 2.0, st.session_state['tolerance_all'])

        if unique_chainages_local:
            # Limit to 20 to avoid memory issues and long rendering times
            for chainage in unique_chainages_local[:20]:
                with st.expander(f"Chainage {chainage:.1f}"):
                    df_dig_loop = pd.DataFrame()
                    if not df_manual_digging.empty and 'Original_chainage' in df_manual_digging.columns:
                        df_dig_loop = df_manual_digging[
                            abs(df_manual_digging['Original_chainage'] - chainage) <= st.session_state['tolerance_all']
                        ]

                    df_leak_loop = pd.DataFrame()
                    if not df_lds_IV.empty and 'chainage' in df_lds_IV.columns:
                        df_leak_loop = df_lds_IV[
                            abs(df_lds_IV['chainage'] - chainage) <= st.session_state['tolerance_all']
                        ]

                    col1, col2 = st.columns(2)
                    col1.metric("Digging", len(df_dig_loop))
                    col2.metric("Leaks", len(df_leak_loop))

                    plot_loop_made = False
                    if (not df_dig_loop.empty and 'DateTime' in df_dig_loop.columns and 'Original_chainage' in df_dig_loop.columns) or \
                       (not df_leak_loop.empty and 'DateTime' in df_leak_loop.columns and 'chainage' in df_leak_loop.columns):

                        plt.close('all')
                        fig_loop, ax_loop = plt.subplots(figsize=(14, 7))

                        if not df_dig_loop.empty:
                            sns.scatterplot(data=df_dig_loop, x='DateTime', y='Original_chainage',
                                          color='blue', marker='o', s=50, ax=ax_loop)
                            plot_loop_made = True
                        if not df_leak_loop.empty:
                            sns.scatterplot(data=df_leak_loop, x='DateTime', y='chainage',
                                          color='red', marker='X', s=80, ax=ax_loop)
                            plot_loop_made = True

                        if plot_loop_made:
                            plt.title(f'Chainage {chainage:.1f} ±{st.session_state["tolerance_all"]}')
                            plt.tight_layout()
                            st.pyplot(fig_loop)
                    else:
                        st.info(f"No digging or leak events found for chainage {chainage:.1f} ±{st.session_state['tolerance_all']}km.")
        else:
            st.info("Cannot perform 'All Unique Chainages Analysis' as no unique chainages were found in the data.")

    st.markdown("---")
    st.caption("💡 Upload files → Adjust chainage → Click Analyze → View correlations!")

# --- LLM Chat Tab ---
with tab3:
    st.subheader("💬 LLM-Powered Leak Analysis Insights")

    # Load GROQ_API_KEY securely from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        st.warning("GROQ_API_KEY environment variable not found. Please set it or enter manually.")
        groq_api_key = st.text_input("Enter your GROQ_API_KEY manually", type="password")

    llm_client = None
    if groq_api_key:
        try:
            llm_client = Groq(api_key=groq_api_key)
            st.success("Groq client initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}. Please check your GROQ_API_KEY.")
    else:
        st.warning("Please provide your GROQ_API_KEY to enable LLM features.")

    # --- Generate Enhanced Analysis Summary for LLM ---
    st.markdown("---")
    st.subheader("Data Summary for LLM Context:")

    if not df_manual_digging.empty or not df_lds_IV.empty:
        total_digging_events = len(df_manual_digging) if not df_manual_digging.empty else 0
        total_leak_events = len(df_lds_IV) if not df_lds_IV.empty else 0

        # These depend on the specific target_chainage and tolerance set in Tab 2
        df_digging_filtered_llm = pd.DataFrame()
        if not df_manual_digging.empty and 'Original_chainage' in df_manual_digging.columns:
            df_digging_filtered_llm = df_manual_digging[
                abs(df_manual_digging['Original_chainage'] - st.session_state['target_chainage']) <= st.session_state['tolerance']
            ].copy()

        df_leaks_filtered_llm = pd.DataFrame()
        if not df_lds_IV.empty and 'chainage' in df_lds_IV.columns:
            df_leaks_filtered_llm = df_lds_IV[
                abs(df_lds_IV['chainage'] - st.session_state['target_chainage']) <= st.session_state['tolerance']
            ].copy()

        filtered_digging_count = len(df_digging_filtered_llm)
        filtered_leak_count = len(df_leaks_filtered_llm)

        first_unique_chainage = "N/A"
        loop_dig_count = 0
        loop_leak_count = 0

        # Ensure unique_chainages is up-to-date from session state
        unique_chainages_for_summary = st.session_state['unique_chainages']
        tolerance_all_for_summary = st.session_state['tolerance_all']

        if unique_chainages_for_summary:
            first_unique_chainage = unique_chainages_for_summary[0]
            if not df_manual_digging.empty and 'Original_chainage' in df_manual_digging.columns:
                df_dig_loop_first = df_manual_digging[
                    abs(df_manual_digging['Original_chainage'] - first_unique_chainage) <= tolerance_all_for_summary
                ]
                loop_dig_count = len(df_dig_loop_first)

            if not df_lds_IV.empty and 'chainage' in df_lds_IV.columns:
                df_leak_loop_first = df_lds_IV[
                    abs(df_lds_IV['chainage'] - first_unique_chainage) <= tolerance_all_for_summary
                ]
                loop_leak_count = len(df_leak_loop_first)

        total_digging_events_all_chains = 0
        total_leak_events_all_chains = 0
        chainage_stats = []

        if unique_chainages_for_summary and not df_manual_digging.empty and not df_lds_IV.empty:
            for chainage_val in unique_chainages_for_summary:
                current_dig_count = 0
                if 'Original_chainage' in df_manual_digging.columns:
                    df_dig_loop_summary = df_manual_digging[
                        abs(df_manual_digging['Original_chainage'] - chainage_val) <= tolerance_all_for_summary
                    ]
                    current_dig_count = len(df_dig_loop_summary)

                current_leak_count = 0
                if 'chainage' in df_lds_IV.columns:
                    df_leak_loop_summary = df_lds_IV[
                        abs(df_lds_IV['chainage'] - chainage_val) <= tolerance_all_for_summary
                    ]
                    current_leak_count = len(df_leak_loop_summary)

                total_digging_events_all_chains += current_dig_count
                total_leak_events_all_chains += current_leak_count
                chainage_stats.append({'chainage': chainage_val, 'digging': current_dig_count, 'leak': current_leak_count})

        avg_digging_per_chainage = total_digging_events_all_chains / len(unique_chainages_for_summary) if unique_chainages_for_summary else 0
        avg_leak_per_chainage = total_leak_events_all_chains / len(unique_chainages_for_summary) if unique_chainages_for_summary else 0

        top_3_leaks = sorted(chainage_stats, key=lambda x: x['leak'], reverse=True)[:3]
        top_3_digging = sorted(chainage_stats, key=lambda x: x['digging'], reverse=True)[:3]

        top_leaks_str = ", ".join([f"Chainage {s['chainage']:.1f} ({s['leak']} leaks)" for s in top_3_leaks])
        top_digging_str = ", ".join([f"Chainage {s['chainage']:.1f} ({s['digging']} diggings)" for s in top_3_digging])

        st.session_state['analysis_summary'] = (
            f"Overall analysis: Total digging events: {total_digging_events:,}, "
            f"Total leak events: {total_leak_events:,}. "
            f"For target chainage {st.session_state['target_chainage']} +/- {st.session_state['tolerance']} km: "
            f"Digging events: {filtered_digging_count}, "
            f"Leak events: {filtered_leak_count}. "
            f"First unique chainage processed ({first_unique_chainage:.1f}) "
            f"had {loop_dig_count} digging events and {loop_leak_count} leak events.\n\n"
            f"Detailed analysis across all {len(unique_chainages_for_summary)} unique chainages (tolerance {tolerance_all_for_summary} km): "
            f"Total digging events in loop: {total_digging_events_all_chains:,}, "
            f"Total leak events in loop: {total_leak_events_all_chains:,}. "
            f"Average digging events per chainage: {avg_digging_per_chainage:.2f}, "
            f"Average leak events per chainage: {avg_leak_per_chainage:.2f}.\n"
            f"Top 3 chainages by leak events: {top_leaks_str}.\n"
            f"Top 3 chainages by digging events: {top_digging_str}."
        )
        st.write(st.session_state['analysis_summary'])
    else:
        st.info("No data loaded to generate analysis summary for LLM context.")
        st.session_state['analysis_summary'] = "No data available to generate an analysis summary."


    # LLM Chat Interface
    user_question = st.text_area(
        "Ask the LLM about the leak detection analysis:",
        placeholder="e.g., 'What are the main correlations I should look for?'",
        height=100
    )

    if st.button("Get LLM Insight", type="primary"):
        if llm_client is None:
            st.error("LLM client not initialized. Please provide your API key.")
        elif not st.session_state['analysis_summary'] or st.session_state['analysis_summary'] == "No data available to generate an analysis summary.":
            st.warning("Analysis summary is empty. Please ensure data is loaded and analysis is performed in the 'Analysis' tab.")
        elif not user_question:
            st.warning("Please enter a question for the LLM.")
        else:
            with st.spinner("Generating insight..."):
                full_prompt = f"""Given the following summary of a leak detection analysis:\n{st.session_state['analysis_summary']}\nUser Question: {user_question}\nBased on the provided analysis summary, please answer the user's question and provide any relevant insights."""

                try:
                    chat_completion = llm_client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": full_prompt}
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.7,
                        max_tokens=1000
                    )
                    st.markdown("### LLM Response:")
                    st.write(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error calling Groq LLM: {e}")
