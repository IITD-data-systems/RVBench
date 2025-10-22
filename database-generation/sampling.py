import pandas as pd
import sys

def sample_wiki_data(x: int, chunksize: int = 100_000):
    # --- Step 1: Page subset (small enough to load fully) ---
    print("start")
    page_df = pd.read_csv("page_csv_files/page.csv", header=None, names=["page_id", "page_title"], nrows=x)
    page_extra_df = pd.read_csv("page_csv_files/page_extra.csv", header=None,
                                names=["page_len", "page_touched", "page_namespace"], nrows=x)
    page_emb_df = pd.read_csv("page_csv_files/embedding.csv", header=None, names=["embedding"], nrows=x)
    print("start2")
    # Save sampled versions (no headers)
    page_df.to_csv(f"page_csv_files/page_{x}.csv", index=False, header=False)
    page_extra_df.to_csv(f"page_csv_files/page_extra_{x}.csv", index=False, header=False)
    page_emb_df.to_csv(f"page_csv_files/embedding_{x}.csv", index=False, header=False)

    page_ids = set(page_df["page_id"].astype(int))

    # --- Step 2: Revision subset ---
    print("start3")
    rev_ids = set()
    with open(f"revision_csv_files/revision_clean_{x}.csv", "w") as rev_out:
        for chunk in pd.read_csv("revision_csv_files/revision_clean.csv", header=None, chunksize=chunksize,
                                 names=["rev_id", "rev_page", "rev_minor_edit", "rev_actor", "rev_timestamp"]):
            filtered = chunk[chunk["rev_page"].isin(page_ids)]
            if not filtered.empty:
                filtered.to_csv(rev_out, index=False, header=False)
                rev_ids.update(filtered["rev_id"].astype(int))

    # --- Step 3: Text subset ---
    print("start4")
    with open(f"text_csv_files/text_{x}.csv", "w") as text_out, \
         open(f"text_csv_files/embedding_{x}.csv", "w") as text_emb_out:

        text_iter = pd.read_csv("text_csv_files/text.csv", header=None, names=["old_id", "old_text"], chunksize=chunksize)
        emb_iter = pd.read_csv("text_csv_files/embedding.csv", header=None, names=["embedding"], chunksize=chunksize)
        count=0
        for chunk_text, chunk_emb in zip(text_iter, emb_iter):
            mask = chunk_text["old_id"].isin(rev_ids)
            filtered_text = chunk_text[mask]
            filtered_emb = chunk_emb[mask]

            if not filtered_text.empty:
                filtered_text.to_csv(text_out, index=False, header=False)
                filtered_emb.to_csv(text_emb_out, index=False, header=False)
            print(count)
            count+=1
    print("start5")
    # --- Step 4: CategoryLinks subset ---
    first_chunk = True
    with open(f"category_csv_files/category_links_clean_{x}.csv", "w") as cat_out:
        for chunk in pd.read_csv("category_csv_files/category_links_clean.csv", header=0,
                                 names=["cl_from", "cl_to"], chunksize=chunksize):
            filtered = chunk[chunk["cl_from"].isin(page_ids)]
            if not filtered.empty:
                filtered.to_csv(cat_out, index=False, header=first_chunk)
                first_chunk = False

    print(f"Sampling complete for {x} rows!")

# Example usage:
if __name__ == "__main__":
    rows = int(sys.argv[1])
    sample_wiki_data(rows)
