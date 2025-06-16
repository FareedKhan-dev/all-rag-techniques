{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "vscode": {
     "languageId": "markdown"
    }
   },
   "source": [
    "# Introduction to Simple RAG\n",
    "\n",
    "Retrieval-Augmented Generation (RAG) is a hybrid approach that combines information retrieval with generative models. It enhances the performance of language models by incorporating external knowledge, which improves accuracy and factual correctness.\n",
    "\n",
    "In a Simple RAG setup, we follow these steps:\n",
    "\n",
    "1. **Data Ingestion**: Load and preprocess the text data.\n",
    "2. **Chunking**: Break the data into smaller chunks to improve retrieval performance.\n",
    "3. **Embedding Creation**: Convert the text chunks into numerical representations using an embedding model.\n",
    "4. **Semantic Search**: Retrieve relevant chunks based on a user query.\n",
    "5. **Response Generation**: Use a language model to generate a response based on retrieved text.\n",
    "\n",
    "This notebook implements a Simple RAG approach, evaluates the model’s response, and explores various improvements."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setting Up the Environment\n",
    "We begin by importing necessary libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import fitz\n",
    "import os\n",
    "import numpy as np\n",
    "import json\n",
    "from openai import OpenAI"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Extracting Text from a PDF File\n",
    "To implement RAG, we first need a source of textual data. In this case, we extract text from a PDF file using the PyMuPDF library."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_text_from_pdf(pdf_path):\n",
    "    \"\"\"\n",
    "    Extracts text from a PDF file and prints the first `num_chars` characters.\n",
    "\n",
    "    Args:\n",
    "    pdf_path (str): Path to the PDF file.\n",
    "\n",
    "    Returns:\n",
    "    str: Extracted text from the PDF.\n",
    "    \"\"\"\n",
    "    # Open the PDF file\n",
    "    mypdf = fitz.open(pdf_path)\n",
    "    all_text = \"\"  # Initialize an empty string to store the extracted text\n",
    "\n",
    "    # Iterate through each page in the PDF\n",
    "    for page_num in range(mypdf.page_count):\n",
    "        page = mypdf[page_num]  # Get the page\n",
    "        text = page.get_text(\"text\")  # Extract text from the page\n",
    "        all_text += text  # Append the extracted text to the all_text string\n",
    "\n",
    "    return all_text  # Return the extracted text"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chunking the Extracted Text\n",
    "Once we have the extracted text, we divide it into smaller, overlapping chunks to improve retrieval accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def chunk_text(text, n, overlap):\n",
    "    \"\"\"\n",
    "    Chunks the given text into segments of n characters with overlap.\n",
    "\n",
    "    Args:\n",
    "    text (str): The text to be chunked.\n",
    "    n (int): The number of characters in each chunk.\n",
    "    overlap (int): The number of overlapping characters between chunks.\n",
    "\n",
    "    Returns:\n",
    "    List[str]: A list of text chunks.\n",
    "    \"\"\"\n",
    "    chunks = []  # Initialize an empty list to store the chunks\n",
    "    \n",
    "    # Loop through the text with a step size of (n - overlap)\n",
    "    for i in range(0, len(text), n - overlap):\n",
    "        # Append a chunk of text from index i to i + n to the chunks list\n",
    "        chunks.append(text[i:i + n])\n",
    "\n",
    "    return chunks  # Return the list of text chunks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setting Up the OpenAI API Client\n",
    "We initialize the OpenAI client to generate embeddings and responses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the OpenAI client with the base URL and API key\n",
    "client = OpenAI(\n",
    "    base_url=\"https://api.studio.nebius.com/v1/\",\n",
    "    api_key=os.getenv(\"OPENAI_API_KEY\")  # Retrieve the API key from environment variables\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Extracting and Chunking Text from a PDF File\n",
    "Now, we load the PDF, extract text, and split it into chunks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of text chunks: 42\n",
      "\n",
      "First text chunk:\n",
      "Understanding Artificial Intelligence \n",
      "Chapter 1: Introduction to Artificial Intelligence \n",
      "Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot \n",
      "to perform tasks commonly associated with intelligent beings. The term is frequently applied to \n",
      "the project of developing systems endowed with the intellectual processes characteristic of \n",
      "humans, such as the ability to reason, discover meaning, generalize, or learn from past \n",
      "experience. Over the past few decades, advancements in computing power and data availability \n",
      "have significantly accelerated the development and deployment of AI. \n",
      "Historical Context \n",
      "The idea of artificial intelligence has existed for centuries, often depicted in myths and fiction. \n",
      "However, the formal field of AI research began in the mid-20th century. The Dartmouth Workshop \n",
      "in 1956 is widely considered the birthplace of AI. Early AI research focused on problem-solving \n",
      "and symbolic methods. The 1980s saw a rise in exp\n"
     ]
    }
   ],
   "source": [
    "# Define the path to the PDF file\n",
    "pdf_path = \"data/AI_Information.pdf\"\n",
    "\n",
    "# Extract text from the PDF file\n",
    "extracted_text = extract_text_from_pdf(pdf_path)\n",
    "\n",
    "# Chunk the extracted text into segments of 1000 characters with an overlap of 200 characters\n",
    "text_chunks = chunk_text(extracted_text, 1000, 200)\n",
    "\n",
    "# Print the number of text chunks created\n",
    "print(\"Number of text chunks:\", len(text_chunks))\n",
    "\n",
    "# Print the first text chunk\n",
    "print(\"\\nFirst text chunk:\")\n",
    "print(text_chunks[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating Embeddings for Text Chunks\n",
    "Embeddings transform text into numerical vectors, which allow for efficient similarity search."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_embeddings(text, model=\"BAAI/bge-en-icl\"):\n",
    "    \"\"\"\n",
    "    Creates embeddings for the given text using the specified OpenAI model.\n",
    "\n",
    "    Args:\n",
    "    text (str): The input text for which embeddings are to be created.\n",
    "    model (str): The model to be used for creating embeddings. Default is \"BAAI/bge-en-icl\".\n",
    "\n",
    "    Returns:\n",
    "    dict: The response from the OpenAI API containing the embeddings.\n",
    "    \"\"\"\n",
    "    # Create embeddings for the input text using the specified model\n",
    "    response = client.embeddings.create(\n",
    "        model=model,\n",
    "        input=text\n",
    "    )\n",
    "\n",
    "    return response  # Return the response containing the embeddings\n",
    "\n",
    "# Create embeddings for the text chunks\n",
    "response = create_embeddings(text_chunks)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performing Semantic Search\n",
    "We implement cosine similarity to find the most relevant text chunks for a user query."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cosine_similarity(vec1, vec2):\n",
    "    \"\"\"\n",
    "    Calculates the cosine similarity between two vectors.\n",
    "\n",
    "    Args:\n",
    "    vec1 (np.ndarray): The first vector.\n",
    "    vec2 (np.ndarray): The second vector.\n",
    "\n",
    "    Returns:\n",
    "    float: The cosine similarity between the two vectors.\n",
    "    \"\"\"\n",
    "    # Compute the dot product of the two vectors and divide by the product of their norms\n",
    "    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def semantic_search(query, text_chunks, embeddings, k=5):\n",
    "    \"\"\"\n",
    "    Performs semantic search on the text chunks using the given query and embeddings.\n",
    "\n",
    "    Args:\n",
    "    query (str): The query for the semantic search.\n",
    "    text_chunks (List[str]): A list of text chunks to search through.\n",
    "    embeddings (List[dict]): A list of embeddings for the text chunks.\n",
    "    k (int): The number of top relevant text chunks to return. Default is 5.\n",
    "\n",
    "    Returns:\n",
    "    List[str]: A list of the top k most relevant text chunks based on the query.\n",
    "    \"\"\"\n",
    "    # Create an embedding for the query\n",
    "    query_embedding = create_embeddings(query).data[0].embedding\n",
    "    similarity_scores = []  # Initialize a list to store similarity scores\n",
    "\n",
    "    # Calculate similarity scores between the query embedding and each text chunk embedding\n",
    "    for i, chunk_embedding in enumerate(embeddings):\n",
    "        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))\n",
    "        similarity_scores.append((i, similarity_score))  # Append the index and similarity score\n",
    "\n",
    "    # Sort the similarity scores in descending order\n",
    "    similarity_scores.sort(key=lambda x: x[1], reverse=True)\n",
    "    # Get the indices of the top k most similar text chunks\n",
    "    top_indices = [index for index, _ in similarity_scores[:k]]\n",
    "    # Return the top k most relevant text chunks\n",
    "    return [text_chunks[index] for index in top_indices]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Running a Query on Extracted Chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Query: What is 'Explainable AI' and why is it considered important?\n",
      "Context 1:\n",
      "systems. Explainable AI (XAI) \n",
      "techniques aim to make AI decisions more understandable, enabling users to assess their \n",
      "fairness and accuracy. \n",
      "Privacy and Data Protection \n",
      "AI systems often rely on large amounts of data, raising concerns about privacy and data \n",
      "protection. Ensuring responsible data handling, implementing privacy-preserving techniques, \n",
      "and complying with data protection regulations are crucial. \n",
      "Accountability and Responsibility \n",
      "Establishing accountability and responsibility for AI systems is essential for addressing potential \n",
      "harms and ensuring ethical behavior. This includes defining roles and responsibilities for \n",
      "developers, deployers, and users of AI systems. \n",
      "Chapter 20: Building Trust in AI \n",
      "Transparency and Explainability \n",
      "Transparency and explainability are key to building trust in AI. Making AI systems understandable \n",
      "and providing insights into their decision-making processes helps users assess their reliability \n",
      "and fairness. \n",
      "Robustness and Reliability \n",
      "\n",
      "=====================================\n",
      "Context 2:\n",
      " incidents. \n",
      "Environmental Monitoring \n",
      "AI-powered environmental monitoring systems track air and water quality, detect pollution, and \n",
      "support environmental protection efforts. These systems provide real-time data, identify \n",
      "pollution sources, and inform environmental policies. \n",
      "Chapter 15: The Future of AI Research \n",
      "Advancements in Deep Learning \n",
      "Continued advancements in deep learning are expected to drive further breakthroughs in AI. \n",
      "Research is focused on developing more efficient and interpretable deep learning models, as well \n",
      "as exploring new architectures and training techniques. \n",
      "Explainable AI (XAI) \n",
      "Explainable AI (XAI) aims to make AI systems more transparent and understandable. Research in \n",
      "XAI focuses on developing methods for explaining AI decisions, enhancing trust, and improving \n",
      "accountability. \n",
      "AI and Neuroscience \n",
      "The intersection of AI and neuroscience is a promising area of research. Understanding the \n",
      "human brain can inspire new AI algorithms and architectures, \n",
      "=====================================\n"
     ]
    }
   ],
   "source": [
    "# Load the validation data from a JSON file\n",
    "with open('data/val.json') as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "# Extract the first query from the validation data\n",
    "query = data[0]['question']\n",
    "\n",
    "# Perform semantic search to find the top 2 most relevant text chunks for the query\n",
    "top_chunks = semantic_search(query, text_chunks, response.data, k=2)\n",
    "\n",
    "# Print the query\n",
    "print(\"Query:\", query)\n",
    "\n",
    "# Print the top 2 most relevant text chunks\n",
    "for i, chunk in enumerate(top_chunks):\n",
    "    print(f\"Context {i + 1}:\\n{chunk}\\n=====================================\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generating a Response Based on Retrieved Chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the system prompt for the AI assistant\n",
    "system_prompt = \"You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'\"\n",
    "\n",
    "def generate_response(system_prompt, user_message, model=\"meta-llama/Llama-3.2-3B-Instruct\"):\n",
    "    \"\"\"\n",
    "    Generates a response from the AI model based on the system prompt and user message.\n",
    "\n",
    "    Args:\n",
    "    system_prompt (str): The system prompt to guide the AI's behavior.\n",
    "    user_message (str): The user's message or query.\n",
    "    model (str): The model to be used for generating the response. Default is \"meta-llama/Llama-2-7B-chat-hf\".\n",
    "\n",
    "    Returns:\n",
    "    dict: The response from the AI model.\n",
    "    \"\"\"\n",
    "    response = client.chat.completions.create(\n",
    "        model=model,\n",
    "        temperature=0,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": system_prompt},\n",
    "            {\"role\": \"user\", \"content\": user_message}\n",
    "        ]\n",
    "    )\n",
    "    return response\n",
    "\n",
    "# Create the user prompt based on the top chunks\n",
    "user_prompt = \"\\n\".join([f\"Context {i + 1}:\\n{chunk}\\n=====================================\\n\" for i, chunk in enumerate(top_chunks)])\n",
    "user_prompt = f\"{user_prompt}\\nQuestion: {query}\"\n",
    "\n",
    "# Generate AI response\n",
    "ai_response = generate_response(system_prompt, user_prompt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluating the AI Response\n",
    "We compare the AI response with the expected answer and assign a score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Based on the evaluation criteria, I would assign a score of 0.8 to the AI assistant's response.\n",
      "\n",
      "The AI assistant's response is very close to the true response, but there are some minor differences. The true response mentions \"transparency\" and \"accountability\" explicitly, which are not mentioned in the AI assistant's response. However, the overall meaning and content of the response are identical, and the AI assistant's response effectively conveys the importance of Explainable AI in building trust and ensuring fairness in AI systems.\n",
      "\n",
      "Therefore, the score of 0.8 reflects the AI assistant's response being very close to the true response, but not perfectly aligned.\n"
     ]
    }
   ],
   "source": [
    "# Define the system prompt for the evaluation system\n",
    "evaluate_system_prompt = \"You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5.\"\n",
    "\n",
    "# Create the evaluation prompt by combining the user query, AI response, true response, and evaluation system prompt\n",
    "evaluation_prompt = f\"User Query: {query}\\nAI Response:\\n{ai_response.choices[0].message.content}\\nTrue Response: {data[0]['ideal_answer']}\\n{evaluate_system_prompt}\"\n",
    "\n",
    "# Generate the evaluation response using the evaluation system prompt and evaluation prompt\n",
    "evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)\n",
    "\n",
    "# Print the evaluation response\n",
    "print(evaluation_response.choices[0].message.content)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv-new-specific-rag",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
