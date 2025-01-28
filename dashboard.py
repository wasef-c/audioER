import pandas as pd
import streamlit as st
import ast
import time

LOG_FILE = "/media/carol/Data/Documents/Emo_rec/NewMel/IEMO_Mel_6/20250122_7/training_logs.csv"

# ----------------------------------------------------------------------
# 1. Helper to slice the logs for a given speaker
# ----------------------------------------------------------------------
def slice_logs_for_speaker(df: pd.DataFrame, speaker_id: str):
    """
    Returns the subset of df that corresponds only to the lines
    after 'STARTING SPEAKER speaker_id' and before the next 'STARTING SPEAKER ...'
    """
    # Find all rows that mark the start of any speaker
    speaker_starts = df[df["log"].str.contains(r"STARTING SPEAKER")].index.to_list()

    # Find the row index for this speaker
    idx_this = df[df["log"].str.contains(f"STARTING SPEAKER {speaker_id}")].index
    if len(idx_this) == 0:
        return pd.DataFrame()  # not found
    start_index = idx_this[0]

    # Find the next start for the *subsequent* speaker (if it exists)
    # so we know where to stop slicing for this speaker
    next_start_index = None
    for i in speaker_starts:
        if i > start_index:
            next_start_index = i
            break
    
    if next_start_index:
        return df.loc[start_index : next_start_index - 1]
    else:
        # If there is no next speaker, slice until the end
        return df.loc[start_index:]


def parse_logs_for_speaker(speaker_df: pd.DataFrame):
    """
    Parse out class_weights, eval, train from the subset of logs
    for *one* speaker only.
    """
    # Class weights
    df_class_weights = speaker_df[speaker_df["log"].str.contains("Class weights")].copy()
    df_class_weights["class_weights"] = (
        df_class_weights["log"]
        .str.extract(r"\[(.*?)\]")
        .apply(lambda x: [float(i) for i in x[0].split()], axis=1)
    )

    # Evaluation logs (eval_uar, eval_accuracy, etc.)
    df_eval = speaker_df[speaker_df["log"].str.contains("'eval_loss'")].copy()
    df_eval["metrics"] = df_eval["log"].apply(ast.literal_eval)
    df_eval = pd.concat(
        [df_eval.drop("metrics", axis=1), df_eval["metrics"].apply(pd.Series)],
        axis=1
    )

    # Training logs
    df_train = speaker_df[speaker_df["log"].str.contains("'grad_norm'")].copy()
    df_train["metrics"] = df_train["log"].apply(ast.literal_eval)
    df_train = pd.concat(
        [df_train.drop("metrics", axis=1), df_train["metrics"].apply(pd.Series)],
        axis=1
    )

    return df_class_weights, df_eval, df_train


def parse_test_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Global test logs (not speaker-specific).
    """
    df_test = df[df["log"].str.contains("_ACC")].copy()
    df_test["metrics"] = df_test["log"].apply(ast.literal_eval)
    df_test = pd.concat(
        [df_test.drop("metrics", axis=1), df_test["metrics"].apply(pd.Series)],
        axis=1
    )
    return df_test


# Initialize Streamlit
st.title("Live Training Dashboard")
st.sidebar.title("Dashboard Settings")
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)

placeholder = st.empty()

# We keep track of which speakers we've already seen and
# store the best eval_uar for each in a dictionary.
# (You could also use st.session_state if desired.)
best_eval_uar_by_speaker = {}
previous_speakers = set()

while True:
    try:
        df = pd.read_csv(LOG_FILE)

        # Grab global test logs once (this will accumulate for all speakers)
        df_test = parse_test_logs(df)

        # Identify all speakers mentioned so far
        df_speakers = df[df["log"].str.contains("STARTING SPEAKER")].copy()
        df_speakers["speaker_id"] = df_speakers["log"].str.extract(r"STARTING SPEAKER (\d+)")
        all_speakers = df_speakers["speaker_id"].unique().tolist()

        # Which new speakers have appeared in the log since last iteration?
        new_speakers = set(all_speakers) - previous_speakers

        # For each newly detected speaker, we'll parse & display fresh metrics
        # (In practice you might only show the *latest* speaker, or all.)
        with placeholder.container():
            for speaker_id in sorted(new_speakers, key=int):
                st.header(f"Metrics for NEW Speaker {speaker_id}")
                speaker_df = slice_logs_for_speaker(df, speaker_id)
                if speaker_df.empty:
                    st.warning(f"No logs found for speaker {speaker_id}.")
                    continue

                # Parse out the training/eval logs for this speaker only
                df_class_weights, df_eval, df_train = parse_logs_for_speaker(speaker_df)

                # ----------------------------------------------------------------
                #  Display the "best eval_uar" from all *previous* speakers
                # ----------------------------------------------------------------
                if best_eval_uar_by_speaker:
                    prev_best = max(best_eval_uar_by_speaker.values())
                    st.write(f"Highest eval_uar so far (previous speakers) = {prev_best:.2f}")
                else:
                    st.write("No previous speakers to compare yet.")

                # ----------------------------------------------------------------
                #  Plot the new speaker’s metrics (epochs start from 0 or 1 again)
                # ----------------------------------------------------------------
                if not df_eval.empty:
                    st.subheader("Eval Metrics (Speaker-Specific)")
                    st.line_chart(
                        df_eval[["epoch", "eval_loss", "eval_accuracy", "eval_uar", "eval_f1"]]
                        .set_index("epoch")
                    )

                    # Find the best epoch by eval_uar for this speaker
                    best_uar_row = df_eval.loc[df_eval["eval_uar"].idxmax()]
                    best_uar = best_uar_row["eval_uar"]
                    best_epoch = best_uar_row["epoch"]
                    st.write(f"**Speaker {speaker_id} best eval_uar** = {best_uar:.2f} at epoch {best_epoch}")
                    
                    # Store it
                    best_eval_uar_by_speaker[speaker_id] = float(best_uar)

                if not df_train.empty:
                    st.subheader("Train Metrics (Speaker-Specific)")
                    st.line_chart(
                        df_train[["epoch", "loss", "learning_rate", "grad_norm"]]
                        .set_index("epoch")
                    )

                if not df_class_weights.empty:
                    st.subheader("Class Weights (Speaker-Specific)")
                    weights_df = pd.DataFrame(
                        df_class_weights["class_weights"].to_list(),
                        columns=["Class 1", "Class 2", "Class 3", "Class 4"]
                    )
                    # Just label them in order: 1..N
                    weights_df["epoch"] = range(1, len(weights_df) + 1)
                    st.line_chart(weights_df.set_index("epoch"))

                # Done with this new speaker
                previous_speakers.add(speaker_id)

            # --------------------------------------------------------------------
            #  Always show the global test metrics (across all speakers so far)
            # --------------------------------------------------------------------
            if not df_test.empty:
                st.subheader("Global Test Metrics Over Time (All Speakers)")
                # In your logs, you might have columns like:
                # IEMO_Mel_6_ACC, IEMO_Mel_6_UAR, MSPI_Mel6_ACC, MSPI_Mel6_UAR, ...
                # Adjust as needed:
                columns_for_test = [
                    col for col in df_test.columns
                    if any(metric in col for metric in ["ACC", "UAR"])  # adjust your filter
                ]
                st.line_chart(df_test[columns_for_test].reset_index(drop=True))

    except FileNotFoundError:
        st.error("Log file not found. Please start training to generate logs.")

    # Sleep until next refresh
    # print(refresh_interval)
    time.sleep(refresh_interval)
