import csv
import random

# Set seed for reproducibility
random.seed(42)

records = []

# Generate Pass records (85 records)
for _ in range(85):
    attendance = random.randint(75, 100)
    internal_marks = random.randint(62, 98)
    study_hours = random.randint(4, 10)
    previous_performance = random.randint(65, 98)
    assignment_score = random.randint(70, 100)
    records.append((attendance, internal_marks, study_hours, previous_performance, assignment_score, "Pass"))

# Generate Fail records (85 records)
for _ in range(85):
    attendance = random.randint(35, 72)
    internal_marks = random.randint(20, 55)
    study_hours = random.randint(1, 4)
    previous_performance = random.randint(25, 55)
    assignment_score = random.randint(25, 55)
    records.append((attendance, internal_marks, study_hours, previous_performance, assignment_score, "Fail"))

# Add 30 overlapping/hard boundary records for realistic model comparison
for _ in range(15):
    # Borderline Pass
    attendance = random.randint(68, 76)
    internal_marks = random.randint(52, 60)
    study_hours = random.randint(3, 5)
    previous_performance = random.randint(52, 62)
    assignment_score = random.randint(55, 65)
    records.append((attendance, internal_marks, study_hours, previous_performance, assignment_score, "Pass"))

for _ in range(15):
    # Borderline Fail (e.g. high attendance but low marks & study hours)
    attendance = random.randint(70, 82)
    internal_marks = random.randint(48, 56)
    study_hours = random.randint(2, 3)
    previous_performance = random.randint(45, 55)
    assignment_score = random.randint(48, 58)
    records.append((attendance, internal_marks, study_hours, previous_performance, assignment_score, "Fail"))

# Shuffle dataset
random.shuffle(records)

# Write to dataset.csv
header = ["attendance", "internal_marks", "study_hours", "previous_performance", "assignment_score", "result"]
csv_path = r"c:\Users\Yugant\OneDrive\Desktop\ML project\Student-Performance-Prediction\dataset.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(records)

print(f"Realistic dataset created successfully at {csv_path} with {len(records)} records!")
