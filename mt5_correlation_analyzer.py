import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import itertools

def main():
    st.title("MT5 Correlation Analyzer")
    st.divider()
    
    st.sidebar.title("File Import")
    st.sidebar.markdown("## Select CSV files")
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, help="Only CSV files are allowed.")
    
    if uploaded_files:
        try:
            dataframes = {}
            all_dates = []

            for uploaded_file in uploaded_files:
                if uploaded_file.name.startswith("testergraph.report."):
                    filename = uploaded_file.name.replace("testergraph.report.", "")
                else:
                    filename = uploaded_file.name

                df = pd.read_csv(uploaded_file, encoding='utf-16', sep='\t')

                if '<DATE>' not in df.columns or '<BALANCE>' not in df.columns:
                    st.error(f"The file {uploaded_file.name} does not contain the '<DATE>' and/or '<BALANCE>' columns.")
                    continue

                df = df.drop(columns=['<EQUITY>', '<DEPOSIT LOAD>'], errors='ignore')
                df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y.%m.%d %H:%M')
                df['DATE_ONLY'] = df['<DATE>'].dt.date
                df = df.drop_duplicates(subset='DATE_ONLY', keep='last').drop(columns=['DATE_ONLY'])
                df['BALANCE_DIFF'] = df['<BALANCE>'].diff()
                df.index = df['<DATE>'].dt.date
                dataframes[filename] = df[['BALANCE_DIFF']]
                all_dates.append(pd.DataFrame(df.index, columns=['<DATE>']))

            all_dates = pd.concat(all_dates).drop_duplicates().sort_values(by='<DATE>').set_index('<DATE>')

            if not dataframes:
                st.error("No valid file was uploaded.")
                return

            combined_corr_all_days = pd.DataFrame(index=dataframes.keys(), columns=dataframes.keys())
            combined_corr_negative_days = pd.DataFrame(index=dataframes.keys(), columns=dataframes.keys())

            for name1, name2 in itertools.combinations(dataframes.keys(), 2):
                df1 = all_dates.join(dataframes[name1], how='left').fillna(0)
                df2 = all_dates.join(dataframes[name2], how='left', rsuffix='_2').fillna(0)

                # Full correlation (all days)
                correlation_all_days = pd.concat([df1['BALANCE_DIFF'], df2['BALANCE_DIFF']], axis=1).corr(method='pearson')
                combined_corr_all_days.loc[name1, name2] = correlation_all_days.iloc[0, 1]
                combined_corr_all_days.loc[name2, name1] = correlation_all_days.iloc[0, 1]

                # Filter negative days
                negative_days_df = pd.concat([df1['BALANCE_DIFF'], df2['BALANCE_DIFF']], axis=1)
                negative_days_df = negative_days_df[(negative_days_df.iloc[:, 0] < 0) | (negative_days_df.iloc[:, 1] < 0)]
                if not negative_days_df.empty:
                    correlation_negative_days = negative_days_df.corr(method='pearson')
                    combined_corr_negative_days.loc[name1, name2] = correlation_negative_days.iloc[0, 1]
                    combined_corr_negative_days.loc[name2, name1] = correlation_negative_days.iloc[0, 1]
                else:
                    combined_corr_negative_days.loc[name1, name2] = np.nan
                    combined_corr_negative_days.loc[name2, name1] = np.nan

            for name in dataframes.keys():
                combined_corr_all_days.loc[name, name] = 1
                combined_corr_negative_days.loc[name, name] = 1

            fig_all_days = plot_correlation_matrix(combined_corr_all_days, "Correlation Matrix - All Days")
            fig_negative_days = plot_correlation_matrix(combined_corr_negative_days, "Correlation Matrix - Negative Days")

            # Button to save the matrices
            if st.button("Save Correlation Matrices"):
                save_matrices_to_pdf(fig_all_days, fig_negative_days)
                st.success("Matrices saved successfully!")

        except Exception as e:
            st.error(f"Error calculating correlation: {e}")

def plot_correlation_matrix(correlation_matrix, title):
    st.write(f"{title}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix.astype(float), cmap='coolwarm', vmin=-0.2, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(correlation_matrix.index, fontsize=10)
    
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            if pd.notnull(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    return fig

def save_matrices_to_pdf(fig_all_days, fig_negative_days):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig_all_days, bbox_inches='tight')
        pdf.savefig(fig_negative_days, bbox_inches='tight')
    buffer.seek(0)
    st.download_button(
        label="Download PDF",
        data=buffer,
        file_name="correlation_matrices.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()


# To run execute the following command line: streamlit run "File path"