from ai21 import AI21Client
from ai21.models.chat import UserMessage
import PyPDF2
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = AI21Client(api_key=API_key)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, max_token_length=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(tokenizer(word)["input_ids"])
        if current_length + word_length > max_token_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def find_most_relevant_chunks(chunks, question, top_n=3):
    question_embedding = embedding_model.encode([question])
    chunk_embeddings = embedding_model.encode(chunks)

    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    most_relevant_indices = np.argsort(similarities)[-top_n:][::-1]

    relevant_chunks = [chunks[idx] for idx in most_relevant_indices]
    return relevant_chunks, similarities[most_relevant_indices]

def ai21_generate_answer(question, context):

    messages = [
        UserMessage(content=f"Use the information provided below to answer the questions at the end. If the answer to the question is not contained in the provided information, say The answer is not in the context.Context information:{context} Question: {question}",)

    ]

    response = client.chat.completions.create(
        model="jamba-1.5-large",
        messages=messages,
        top_p=1.0
    )
    return response.choices[0].message.content  

def answer_question_from_pdf(pdf_path, question):
    context = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(context, max_token_length=1000)

    most_relevant_chunks, similarities = find_most_relevant_chunks(chunks, question, top_n=3)
    combined_context = " ".join(most_relevant_chunks)

    print(f"Relevant chunks (similarities: {', '.join(map(str, similarities))}):")
    for idx, chunk in enumerate(most_relevant_chunks):
        print(f"Chunk {idx+1}:\n{chunk}...\n")  

    final_answer = ai21_generate_answer(question, combined_context)
    return final_answer

while True:
    pdf_path = "/home/khushi/ai21/data/eco1.pdf"  

    question = input("Enter your question here:")
    if question.lower() == "exit":
        print("Exiting")
        break

    final_answer = answer_question_from_pdf(pdf_path, question)
    print("\nFinal Answer:", final_answer)

