import pandas as pd
import random

# Define some example questions and their corresponding possible answers
qa_pairs = {
    "Why did the men hide?": [
        "The men hid to avoid getting caught.",
        "They were hiding because they were scared.",
        "The men hid to surprise their friend."
    ],
    "What does the woman think?": [
        "The woman thinks something is wrong.",
        "She believes that they might be late.",
        "The woman thinks they should be cautious."
    ],
    "Why did the driver lock Harold in the van?": [
        "The driver locked Harold to keep him safe.",
        "He was locked in by mistake.",
        "The driver didn't realize Harold was still inside."
    ],
    "What is the deliveryman feeling and why?": [
        "The deliveryman is frustrated because he's running late.",
        "He's happy because he finished all his deliveries.",
        "The deliveryman is tired after a long day."
    ],
    "Why did Harold pick up the cat?": [
        "Harold picked up the cat to keep it from running away.",
        "He thought the cat looked lost.",
        "He picked it up because it was blocking his way."
    ],
    "Why does Harold fan Mildred?": [
        "Harold fans Mildred because she feels hot.",
        "He's trying to comfort her.",
        "Harold is fanning her to keep the flies away."
    ]
}

# Generate a simulated dataset
data = {
    "ID": range(1, 101),
    "Question": [],
    "Answer": [],
    "Score": [random.randint(0, 2) for _ in range(100)],  # Assuming a 3-class system (0, 1, 2)
    "Age": [random.randint(8, 13) for _ in range(100)],  # Example ages
    "Gender": [random.choice(["Boy", "Girl"]) for _ in range(100)],
    "Orig_ID": [None] * 100  # Optional, useful for linking augmented data
}

# Populate the questions and answers
for _ in range(100):
    question = random.choice(list(qa_pairs.keys()))
    answer = random.choice(qa_pairs[question])
    data["Question"].append(question)
    data["Answer"].append(answer)

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data)
df.to_csv("Data/original_data.csv", index=False)

print("Sample data generated and saved as 'Data/original_data.csv'")
