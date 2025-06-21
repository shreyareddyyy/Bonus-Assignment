from transformers import pipeline

# Initialize the pipeline with a custom pretrained model
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
Hyderabad is one of the largest IT hubs in India.
It is famous for its biryani, the Charminar monument, and pearl jewelry.
The city is located in the southern part of India and is the capital of Telangana state.
The Hussain Sagar lake, built by Ibrahim Quli Qutb Shah, connects the twin cities of Hyderabad and Secunderabad.
"""

# List of multiple relevant questions
questions = [
    "What is Hyderabad famous for?",
    "Where is Hyderabad located?",
    "What connects Hyderabad and Secunderabad?",
    "Which lake is associated with Hyderabad?",
    "Which state has Hyderabad as its capital?"
]

# Loop through the questions and print the results
for question in questions:
    result = qa(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} | Confidence: {result['score']:.2f}\n")
