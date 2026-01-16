import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# --- Streamlit layout / title ---
st.title("🔍 Digging vs. Leak Events Visualisation")

# --- Enhanced data loading with file upload support ---
uploaded_files = st.file_uploader(
    "📁 Upload Excel files (df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx)",
    type=['xlsx'], accept_multiple_files=True
)

# Try repo files first, fallback to uploads
try:
    df_pidws = pd.read_excel("df_pidws.xlsx")
    df_manual_digging = pd.read_excel("df_manual_digging.xlsx")
    df_lds_IV = pd.read_excel("df_lds_IV.xlsx")
    st.success("✅ Loaded Excel files from repo")
except FileNotFoundError:
    st.warning("⚠️ Excel files not found in repo. Please upload them using the uploader above.")
    if uploaded_files:
        file_dict = {f.name: f for f in uploaded_files}
        
        if 'df_pidws.xlsx' in file_dict:
            df_pidws = pd.read_excel(file_dict['df_pidws.xlsx'])
        else:
            df_pidws = pd.DataFrame()
            st.warning("Missing df_pidws.xlsx")
            
        if 'df_manual_digging.xlsx' in file_dict:
            df_manual_digging = pd.read_excel(file_dict['df_manual_digging.xlsx'])
        else:
            df_manual_digging = pd.DataFrame()
            st.warning("Missing df_manual_digging.xlsx")
            
        if 'df_lds_IV.xlsx' in file_dict:
            df_lds_IV = pd.read_excel(file_dict['df_lds_IV.xlsx'])
        else:
            df_lds_IV = pd.DataFrame()
            st.warning("Missing df_lds_IV.xlsx")
            
        st.success("✅ Loaded uploaded Excel files")
    else:
        st.stop()

# Check if we have required data
if df_manual_digging.empty or df_lds_IV.empty:
    st.error("❌ Need df_manual_digging.xlsx and df_lds_IV.xlsx to proceed.")
    st.stop()

st.write("Imported matplotlib.pyplot as plt and seaborn as sns.")

# --- Your original data prep (unchanged) ---
df_selected_columns = df_manual_digging[['Event Duration', 'Original_chainage', 'DateTime']].copy()
st.write("Preview of selected columns from manual digging data:")
st.dataframe(df_selected_columns.head())

df_manual_digging['Date_new'] = df_manual_digging['DateTime'].dt.date
df_manual_digging['Time_new'] = df_manual_digging['DateTime'].dt.time
df_selected_columns['Date_new'] = df_manual_digging['DateTime'].dt.date
df_selected_columns['Time_new'] = df_manual_digging['DateTime'].dt.time
df_selected_columns = df_selected_columns[['Event Duration', 'Original_chainage', 'Date_new', 'Time_new']].copy()

st.write("Preview of LDS IV data:")
st.dataframe(df_lds_IV.head())

# --- Streamlit number input (replaces input()) ---
col1, col2 = st.columns([3, 1])
with col1:
    target_chainage = st.number_input(
        "🎯 Enter the 'Original_chainage' value you wish to analyze (e.g., 25.4):",
        value=25.4,
        format="%.3f"
    )
with col2:
    tolerance = st.number_input("Tolerance (±km):", value=1.0, min_value=0.1, step=0.1)

st.write(f"📊 Analyzing chainage: **{target_chainage}** (tolerance: **{tolerance}**)")

# --- Data processing (your exact code) ---
df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
st.write("✅ Converted 'Date' column in df_lds_IV to datetime objects.")

df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
st.write("✅ Converted 'Time' column in df_lds_IV to timedelta objects.")

df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
st.write("✅ Created 'DateTime' column in df_lds_IV by combining 'Date' and 'Time'.")

df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage) <= tolerance]
st.write(f"🔍 Filtered digging events: **{len(df_digging_filtered)}** events")

df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage) <= tolerance]
st.write(f"🔍 Filtered leak events: **{len(df_leaks_filtered)}** events")

# --- First scatter plot (your exact visualization) ---
if not df_digging_filtered.empty or not df_leaks_filtered.empty:
    fig1, ax = plt.subplots(figsize=(18, 10))

    # Plotting digging events
    if not df_digging_filtered.empty:
        sns.scatterplot(data=df_digging_filtered, x='DateTime', y='Original_chainage',
                       color='blue', label='Digging Events', marker='o', s=50, ax=ax)

    # Plotting leak events
    if not df_leaks_filtered.empty:
        sns.scatterplot(data=df_leaks_filtered, x='DateTime', y='chainage',
                       color='red', label='Leak Events', marker='X', s=80, ax=ax)

    plt.title(f'Digging vs. Leak Events at Chainage {target_chainage} (Tolerance: {tolerance})', fontsize=16)
    plt.xlabel('Date and Time', fontsize=12)
    plt.ylabel('Chainage', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    st.pyplot(fig1)
    st.success("📈 Generated scatter plot for selected chainage")
else:
    st.warning("No events found within tolerance. Try increasing tolerance.")

# --- Unique chainages analysis (your loop, Streamlit-ified) ---
if st.checkbox("🔄 Analyze ALL unique chainages"):
    unique_chainages = df_manual_digging['Original_chainage'].unique().tolist()
    st.write(f"📋 Found **{len(unique_chainages)}** unique chainage values")
    st.write(f"First 10: {unique_chainages[:10]}")
    
    for i, target_chainage_val in enumerate(unique_chainages):
        with st.expander(f"Chainage {target_chainage_val:.1f} ({i+1}/{len(unique_chainages)})", expanded=False):
            df_digging_filtered_loop = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage_val) <= tolerance]
            df_leaks_filtered_loop = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage_val) <= tolerance]

            if not df_digging_filtered_loop.empty or not df_leaks_filtered_loop.empty:
                fig_loop, ax_loop = plt.subplots(figsize=(16, 8))

                if not df_digging_filtered_loop.empty:
                    sns.scatterplot(data=df_digging_filtered_loop, x='DateTime', y='Original_chainage',
                                   color='blue', label='Digging Events', marker='o', s=50, ax=ax_loop)

                if not df_leaks_filtered_loop.empty:
                    sns.scatterplot(data=df_leaks_filtered_loop, x='DateTime', y='chainage',
                                   color='red', label='Leak Events', marker='X', s=80, ax=ax_loop)

                plt.title(f'Digging vs. Leak Events at Chainage {target_chainage_val:.1f} (Tolerance: {tolerance:.1f})')
                plt.xlabel('Date and Time')
                plt.ylabel('Chainage')
                plt.grid(True, alpha=0.3)
                plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                st.pyplot(fig_loop)
                col1, col2 = st.columns(2)
                col1.metric("Digging Events", len(df_digging_filtered_loop))
                col2.metric("Leak Events", len(df_leaks_filtered_loop))
            else:
                st.info(f"No events found for chainage {target_chainage_val:.1f}")
