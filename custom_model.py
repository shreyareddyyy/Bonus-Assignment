from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = "Charles Babbage is considered the father of the computer."

result = qa(question="Who is the father of the computer?", context=context)
print(result)
