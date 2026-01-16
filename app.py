# --- Enhanced data loading with file upload support ---
uploaded_files = st.file_uploader(
    "Upload Excel files (df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx)",
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
        df_pidws = pd.read_excel(file_dict['df_pidws.xlsx']) if 'df_pidws.xlsx' in file_dict else pd.DataFrame()
        df_manual_digging = pd.read_excel(file_dict['df_manual_digging.xlsx']) if 'df_manual_digging.xlsx' in file_dict else pd.DataFrame()
        df_lds_IV = pd.read_excel(file_dict['df_lds_IV.xlsx']) if 'df_lds_IV.xlsx' in file_dict else pd.DataFrame()
        st.success("✅ Loaded uploaded Excel files")
    else:
        st.stop()

# Rest of your code continues exactly the same...
