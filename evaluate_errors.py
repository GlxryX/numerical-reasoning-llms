import json
import re

with open('results/zeroshot_gpt2.json', 'r') as f:
    data = json.load(f)

def get_cat(item):
    pred = item.get('predicted_answer')
    gold_str = item.get('gold_answer', '')
    match = re.search(r'####\s*(-?\d+\.?\d*)', gold_str)
    gold = float(match.group(1)) if match else None
    
    question = item.get('question', '')
    raw = item.get('raw_output', '')
    
    # Extracted numbers from question
    q_nums = [float(x) for x in re.findall(r'\d+\.?\d*', question)]
    
    if pred is None:
        return "Other"
        
    # Check for "Other" - no valid number really: just counting or no real reasoning output
    # If the output is just a long list of numbers or doesn't have an answer structure
    # Actually just check if it's a huge sequence
    if "1." in raw and "2." in raw and "3." in raw and "4." in raw:
        return "Other"
        
    # Check if pred matches a number in the question
    if pred in q_nums:
        return "Quantity misinterpretation"
        
    if gold is not None:
        diff = abs(pred - gold) / (abs(gold) if gold != 0 else 1)
        if diff <= 0.25:
            # Note: might be exactly correct, but we only have "Arithmetic error" for diff <= 25%
            return "Arithmetic error"
        else:
            return "Multi-step reasoning failure"
    
    return "Other"

for item in data:
    item['manual_error_category'] = get_cat(item)

with open('results/zeroshot_gpt2.json', 'w') as f:
    json.dump(data, f, indent=4)
