import pandas as pd
from datasets import load_dataset
import random

if __name__ == "__main__":
    dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K", split="train")
    print(f"Total dataset length: {len(dataset)}")    
    random_subset = dataset.select(range(10000))

    if "image" in random_subset.column_names:
        random_subset = random_subset.remove_columns("image")
        
    df = pd.DataFrame(random_subset)
    output_csv_path = "coco_30k_random_10k.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"Saved subset to {output_csv_path}")