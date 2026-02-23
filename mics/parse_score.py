import os
import re
import csv


pattern_case = re.compile(r"Case#:\s*(\d+)\:\s*target prompt:\s*(.*)")
pattern_toxic = re.compile(r"toxicity pred:\s*([\d.]+)")


root_path = "results/nudity/vanilla/sld_MAX_SDv1-4_mma-diffusion"

input_log_file = os.path.join(root_path, "logs.txt")
output_csv_file = os.path.join(root_path, "parsed_logs.csv")

results = []


temp_case = None
temp_prompt = None

with open(input_log_file, "r", encoding="utf-8") as f:
    for line in f:
        
        match_case = pattern_case.search(line)
        if match_case:
            
            temp_case = match_case.group(1)    
            temp_prompt = match_case.group(2)  
        else:
            
            match_toxic = pattern_toxic.search(line)
            if match_toxic and temp_case is not None:
                
                toxic_val = match_toxic.group(1)
                results.append([temp_case, temp_prompt, toxic_val])

                
                temp_case = None
                temp_prompt = None


with open(output_csv_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(["Case ID", "Prompt", "Toxicity Pred"])

    
    for row in results:
        writer.writerow(row)

print(f"Result'{output_csv_file}' Saved.")