import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Streamlit layout / title (added, logic unchanged) ---
st.title("Digging vs. Leak Events Visualisation")

st.write("Imported matplotlib.pyplot as plt and seaborn as sns.")

# --- Load data (same as your code) ---
df_pidws = pd.read_excel("df_pidws.xlsx")
df_manual_digging = pd.read_excel("df_manual_digging.xlsx")
df_selected_columns = df_manual_digging[['Event Duration', 'Original_chainage', 'DateTime']].copy()
st.write("Preview of selected columns from manual digging data:")
st.dataframe(df_selected_columns.head())

df_manual_digging['Date_new'] = df_manual_digging['DateTime'].dt.date
df_manual_digging['Time_new'] = df_manual_digging['DateTime'].dt.time
df_selected_columns['Date_new'] = df_manual_digging['DateTime'].dt.date
df_selected_columns['Time_new'] = df_manual_digging['DateTime'].dt.time
df_selected_columns = df_selected_columns[['Event Duration', 'Original_chainage', 'Date_new', 'Time_new']].copy()

df_lds_IV = pd.read_excel("df_lds_IV.xlsx")
st.write("Preview of LDS IV data:")
st.dataframe(df_lds_IV.head())

# --- Replace input() with Streamlit widget (syntax change only) ---
target_chainage = st.number_input(
    "Please enter the 'Original_chainage' value you wish to analyze (e.g., 25.4):",
    value=25.4,
    format="%.3f"
)
st.write(f"User provided target chainage: {target_chainage}")

df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
st.write("Converted 'Date' column in df_lds_IV to datetime objects.")

df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
st.write("Converted 'Time' column in df_lds_IV to timedelta objects.")

df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
st.write("Created 'DateTime' column in df_lds_IV by combining 'Date' and 'Time'.")

tolerance = 1.0
st.write(f"Defined chainage tolerance: {tolerance}")

df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage) <= tolerance]
st.write(f"Filtered df_manual_digging for chainage {target_chainage} with tolerance {tolerance}. Number of events: {len(df_digging_filtered)}")

df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage) <= tolerance]
st.write(f"Filtered df_lds_IV for chainage {target_chainage} with tolerance {tolerance}. Number of events: {len(df_leaks_filtered)}")

# --- First scatter plot (same plotting code, only wrapped for Streamlit) ---
fig1 = plt.figure(figsize=(18, 10))

# Plotting digging events
sns.scatterplot(data=df_digging_filtered, x='DateTime', y='Original_chainage',
                color='blue', label='Digging Events', marker='o', s=50)

# Plotting leak events
sns.scatterplot(data=df_leaks_filtered, x='DateTime', y='chainage',
                color='red', label='Leak Events', marker='X', s=80)

plt.title(f'Digging vs. Leak Events at Chainage {target_chainage} (Tolerance: {tolerance})')
plt.xlabel('Date and Time')
plt.ylabel('Chainage')
plt.grid(True)
plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(right=0.75)

st.pyplot(fig1)
st.write("Generated scatter plot showing digging and leak events for the selected chainage.")

# --- Unique chainages loop (same logic, per-figure display in Streamlit) ---
unique_chainages = df_manual_digging['Original_chainage'].unique().tolist()
st.write(f"Extracted {len(unique_chainages)} unique chainage values.")
st.write(f"First 10 unique chainages: {unique_chainages[:10]}")

for target_chainage_val in unique_chainages:
    df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage_val) <= tolerance]
    df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage_val) <= tolerance]

    if not df_digging_filtered.empty or not df_leaks_filtered.empty:
        fig = plt.figure(figsize=(18, 10))

        if not df_digging_filtered.empty:
            sns.scatterplot(data=df_digging_filtered, x='DateTime', y='Original_chainage',
                            color='blue', label='Digging Events', marker='o', s=50)

        if not df_leaks_filtered.empty:
            sns.scatterplot(data=df_leaks_filtered, x='DateTime', y='chainage',
                            color='red', label='Leak Events', marker='X', s=80)

        plt.title(f'Digging vs. Leak Events at Chainage {target_chainage_val:.1f} (Tolerance: {tolerance:.1f})')
        plt.xlabel('Date and Time')
        plt.ylabel('Chainage')
        plt.grid(True)
        plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.75)

        st.pyplot(fig)
        st.write(f"Processed chainage {target_chainage_val:.1f}: Digging events: {len(df_digging_filtered)}, Leak events: {len(df_leaks_filtered)}")
    else:
        st.write(f"No events found for chainage {target_chainage_val:.1f} with tolerance {tolerance:.1f}. Skipping plot.")
