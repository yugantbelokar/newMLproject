import urllib.request
import urllib.parse
import re
import sys

# Ensure UTF-8 output encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

url = "http://127.0.0.1:5000/predict"

test_cases = [
    {
        "name": "Test Case 1 (Good Student - Expected PASS)",
        "data": {
            "attendance": "85",
            "internal_marks": "72",
            "study_hours": "4",
            "previous_performance": "70",
            "assignment_score": "80"
        }
    },
    {
        "name": "Test Case 2 (Low Performance Student - Expected FAIL)",
        "data": {
            "attendance": "50",
            "internal_marks": "35",
            "study_hours": "1",
            "previous_performance": "40",
            "assignment_score": "45"
        }
    },
    {
        "name": "Test Case 3 (High Performing Student - Expected PASS)",
        "data": {
            "attendance": "95",
            "internal_marks": "90",
            "study_hours": "7",
            "previous_performance": "88",
            "assignment_score": "92"
        }
    }
]

print("==================================================")
print("     FLASK APP PREDICTION API TEST SUITE")
print("==================================================\n")

for tc in test_cases:
    print(f"--- {tc['name']} ---")
    print(f"Inputs: {tc['data']}")
    
    encoded_data = urllib.parse.urlencode(tc['data']).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_data, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            
            # Extract result badge
            result_match = re.search(r'class="result-badge (pass|fail)">\s*([\s\S]*?)</div>', html)
            prob_match = re.search(r'class="probability-value">\s*([\d\.]+)%', html)
            
            result_str = result_match.group(2).strip() if result_match else "Unknown"
            prob_str = prob_match.group(1) if prob_match else "Unknown"
            
            print(f"-> Outcome     : {result_str}")
            print(f"-> Pass Prob.  : {prob_str}%\n")
    except Exception as e:
        print(f"-> Error: {e}\n")

print("==================================================")
print("           ALL TESTS COMPLETED!")
print("==================================================")
