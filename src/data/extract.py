import pandas as pd
import re
import os

RAW_PATH = os.path.join("data", "raw", "twcs.csv")
OUTPUT_PATH = os.path.join("data", "processed", "spotify_threads.csv")

def fix_encoding(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def clean_text(text: str) -> str:
    text = fix_encoding(text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_all_tweets(path: str) -> pd.DataFrame:
    print("Loading full CSV in chunks...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000, encoding="utf-8",
                              low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total tweets loaded: {len(df)}")
    return df

def reconstruct_threads(df: pd.DataFrame) -> pd.DataFrame:
    print("Reconstructing SpotifyCares conversation pairs...")

    # Separate Spotify replies and customer tweets
    spotify_replies = df[df["author_id"] == "SpotifyCares"].copy()
    customer_tweets = df[df["inbound"] == True].copy()

    print(f"SpotifyCares reply tweets: {len(spotify_replies)}")

    # Build a lookup: tweet_id -> row for customer tweets
    # Ensure tweet_id is int for consistent matching
    customer_tweets["tweet_id"] = pd.to_numeric(
        customer_tweets["tweet_id"], errors="coerce"
    ).astype("Int64")
    customer_index = customer_tweets.set_index("tweet_id")

    # For each customer tweet, check if its response_tweet_id
    # points to a SpotifyCares tweet
    # response_tweet_id can be comma-separated — expand it
    spotify_reply_ids = set(
        pd.to_numeric(spotify_replies["tweet_id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    threads = []

    for _, customer_row in customer_tweets.iterrows():
        response_ids_raw = customer_row.get("response_tweet_id")

        if pd.isna(response_ids_raw):
            continue

        # Parse comma-separated response IDs
        try:
            response_ids = [
                int(x.strip())
                for x in str(response_ids_raw).split(",")
                if x.strip().isdigit()
            ]
        except Exception:
            continue

        # Check if any of the responses is a SpotifyCares tweet
        matched_spotify_ids = [
            rid for rid in response_ids if rid in spotify_reply_ids
        ]

        if not matched_spotify_ids:
            continue

        # Take the first matched Spotify reply
        spotify_reply_id = matched_spotify_ids[0]
        spotify_row = spotify_replies[
            spotify_replies["tweet_id"] == spotify_reply_id
        ]

        if spotify_row.empty:
            continue

        spotify_row = spotify_row.iloc[0]

        customer_text = clean_text(str(customer_row["text"]))
        spotify_text = clean_text(str(spotify_row["text"]))

        if not customer_text or not spotify_text:
            continue
        if len(customer_text) < 10:
            continue

        threads.append({
            "thread_id": int(customer_row["tweet_id"]),
            "customer_tweet_id": int(customer_row["tweet_id"]),
            "spotify_reply_id": spotify_reply_id,
            "customer_text": customer_text,
            "spotify_reply": spotify_text,
            "created_at": spotify_row["created_at"]
        })

    threads_df = pd.DataFrame(threads)
    threads_df = threads_df.drop_duplicates(subset=["customer_tweet_id"])

    print(f"Valid conversation pairs found: {len(threads_df)}")
    return threads_df

def main():
    df = load_all_tweets(RAW_PATH)
    threads_df = reconstruct_threads(df)

    if len(threads_df) == 0:
        print("ERROR: Still zero pairs. Printing sample customer tweet structure:")
        sample = df[df["inbound"] == True].head(5)
        print(sample[["tweet_id", "response_tweet_id", "text"]].to_string())
        return

    threads_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nSample output:")
    print(threads_df[["customer_text", "spotify_reply"]].head(3).to_string())

if __name__ == "__main__":
    main()