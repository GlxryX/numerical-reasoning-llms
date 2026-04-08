import re

def extract_predicted_answer(text):
    """
    Assumes the model's final answer is the last number in its output string
    """
    numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text)
    
    if numbers:
        final_number_str = numbers[-1].replace(",", "") # remove commas from last number
        try:
            return float(final_number_str)
        except ValueError:
            return None
    return None

def extract_ground_truth(text):
    """
    GSM8K separates the reasoning and the final answer with '####' in ground truth format.
    """
    if "####" in text:
        ans_str = text.split("####")[-1].strip()
        ans_str = ans_str.replace(",", "")
        try:
            return float(ans_str)
        except ValueError:
            return None
    return None

def exact_match_accuracy(predictions, ground_truths):
    """
    Calculates the exact match accuracy across a list of predictions and truths
    """
    if not predictions or len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must be the same length.")

    correct = 0
    for p, g in zip(predictions, ground_truths):
        p_val = extract_predicted_answer(p)
        g_val = extract_ground_truth(g)
        
        tolerance = 1e-5 # tolerance for floating point comparisons
        if p_val is not None and g_val is not None and abs(p_val - g_val) < tolerance:
            correct += 1
            
    return correct / len(predictions)