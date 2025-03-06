import csv
from faker import Faker
from random import choice, randint
from datetime import datetime, timedelta

# Create an instance of Faker for Thai locale
fake = Faker('th_TH')  # Use Thai locale for names

# Function to calculate age from date of birth
def calculate_age(dob):
    today = datetime.today()
    dob = datetime.strptime(dob, "%Y-%m-%d")
    age = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        age -= 1
    return age

# Function to generate health-related data
def generate_health_data():
    # Generate personal information with Thai names
    first_name = fake.first_name()  # Thai first name
    last_name = fake.last_name()  # Thai last name
    weight = randint(45, 100)  # Weight (kg)
    height = randint(150, 190)  # Height (cm)
    birthdate = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d")  # Date of birth
    age = calculate_age(birthdate)  # Calculate age
    blood_type = choice(["A", "B", "AB", "O"])  # Blood type
    chronic_disease = choice(["Diabetes", "High blood pressure", "Heart disease", "None"])  # Chronic disease
    surgery_history = choice(["Appendectomy", "Heart surgery", "None", "Cancer surgery"])  # Surgery history
    exercise_habits = choice(["Exercising every day", "Exercising 3 times a week", "No exercise"])  # Exercise habits
    diet_habits = choice(["Low fat diet", "Eating vegetables and fruits", "High fat diet"])  # Dietary habits
    smoking = choice(["Smokes", "Does not smoke"])  # Smoking habit
    alcohol = choice(["Drinks alcohol 1 time/week", "Drinks alcohol 3 times/month", "Does not drink alcohol"])  # Alcohol consumption
    stress = choice(["Stress from work", "Stress from studies", "No stress"])  # Stress level
    sleep = choice(["Sleeps 6 hours/day", "Sleeps 8 hours/day", "Sleeps less than 6 hours"])  # Sleep habits
    blood_sugar = choice(["Normal", "High", "Low"])  # Blood sugar level
    cholesterol = choice(["High", "Normal", "Low"])  # Cholesterol level

    # Return all generated data
    return [first_name, last_name, weight, height, birthdate, age, blood_type, chronic_disease, surgery_history,
            exercise_habits, diet_habits, smoking, alcohol, stress, sleep, blood_sugar, cholesterol, "Other information"]

# Generate data for 10,000 people
num_records = 10000
data = []

# Add header row with underscores
data.append(["first_name", "last_name", "weight", "height", "date_of_birth", "age", "blood_type", "chronic_disease", 
             "surgery_history", "exercise_habits", "dietary_habits", "smoking_habit", "alcohol_consumption", 
             "stress_level", "sleep_habits", "blood_sugar_level", "cholesterol_level", "other_information"])

# Generate data for 10,000 people
for _ in range(num_records):
    data.append(generate_health_data())

# Write data to CSV file
with open('health_data_10000.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file with 10,000 entries has been successfully created!")
