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
    "## Evaluating Chunk Sizes in Simple RAG\n",
    "\n",
    "Choosing the right chunk size is crucial for improving retrieval accuracy in a Retrieval-Augmented Generation (RAG) pipeline. The goal is to balance retrieval performance with response quality.\n",
    "\n",
    "This section evaluates different chunk sizes by:\n",
    "\n",
    "1. Extracting text from a PDF.\n",
    "2. Splitting text into chunks of varying sizes.\n",
    "3. Creating embeddings for each chunk.\n",
    "4. Retrieving relevant chunks for a query.\n",
    "5. Generating a response using retrieved chunks.\n",
    "6. Evaluating faithfulness and relevancy.\n",
    "7. Comparing results for different chunk sizes."
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
    "## Extracting Text from the PDF\n",
    "First, we will extract text from the `AI_Information.pdf` file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Understanding Artificial Intelligence \n",
      "Chapter 1: Introduction to Artificial Intelligence \n",
      "Artificial intelligence (AI) refers to the ability of a digital computer or computer-controlled robot \n",
      "to perform tasks commonly associated with intelligent beings. The term is frequently applied to \n",
      "the project of developing systems endowed with the intellectual processes characteristic of \n",
      "humans, such as the ability to reason, discover meaning, generalize, or learn from past \n",
      "experience. Over the past f\n"
     ]
    }
   ],
   "source": [
    "def extract_text_from_pdf(pdf_path):\n",
    "    \"\"\"\n",
    "    Extracts text from a PDF file.\n",
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
    "    \n",
    "    # Iterate through each page in the PDF\n",
    "    for page in mypdf:\n",
    "        # Extract text from the current page and add spacing\n",
    "        all_text += page.get_text(\"text\") + \" \"\n",
    "\n",
    "    # Return the extracted text, stripped of leading/trailing whitespace\n",
    "    return all_text.strip()\n",
    "\n",
    "# Define the path to the PDF file\n",
    "pdf_path = \"data/AI_Information.pdf\"\n",
    "\n",
    "# Extract text from the PDF file\n",
    "extracted_text = extract_text_from_pdf(pdf_path)\n",
    "\n",
    "# Print the first 500 characters of the extracted text\n",
    "print(extracted_text[:500])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chunking the Extracted Text\n",
    "To improve retrieval, we split the extracted text into overlapping chunks of different sizes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chunk Size: 128, Number of Chunks: 326\n",
      "Chunk Size: 256, Number of Chunks: 164\n",
      "Chunk Size: 512, Number of Chunks: 82\n"
     ]
    }
   ],
   "source": [
    "def chunk_text(text, n, overlap):\n",
    "    \"\"\"\n",
    "    Splits text into overlapping chunks.\n",
    "\n",
    "    Args:\n",
    "    text (str): The text to be chunked.\n",
    "    n (int): Number of characters per chunk.\n",
    "    overlap (int): Overlapping characters between chunks.\n",
    "\n",
    "    Returns:\n",
    "    List[str]: A list of text chunks.\n",
    "    \"\"\"\n",
    "    chunks = []  # Initialize an empty list to store the chunks\n",
    "    for i in range(0, len(text), n - overlap):\n",
    "        # Append a chunk of text from the current index to the index + chunk size\n",
    "        chunks.append(text[i:i + n])\n",
    "    \n",
    "    return chunks  # Return the list of text chunks\n",
    "\n",
    "# Define different chunk sizes to evaluate\n",
    "chunk_sizes = [128, 256, 512]\n",
    "\n",
    "# Create a dictionary to store text chunks for each chunk size\n",
    "text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}\n",
    "\n",
    "# Print the number of chunks created for each chunk size\n",
    "for size, chunks in text_chunks_dict.items():\n",
    "    print(f\"Chunk Size: {size}, Number of Chunks: {len(chunks)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating Embeddings for Text Chunks\n",
    "Embeddings convert text into numerical representations for similarity search."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Generating Embeddings: 100%|██████████| 3/3 [00:11<00:00,  3.71s/it]\n"
     ]
    }
   ],
   "source": [
    "from tqdm import tqdm\n",
    "\n",
    "def create_embeddings(texts, model=\"BAAI/bge-en-icl\"):\n",
    "    \"\"\"\n",
    "    Generates embeddings for a list of texts.\n",
    "\n",
    "    Args:\n",
    "    texts (List[str]): List of input texts.\n",
    "    model (str): Embedding model.\n",
    "\n",
    "    Returns:\n",
    "    List[np.ndarray]: List of numerical embeddings.\n",
    "    \"\"\"\n",
    "    # Create embeddings using the specified model\n",
    "    response = client.embeddings.create(model=model, input=texts)\n",
    "    # Convert the response to a list of numpy arrays and return\n",
    "    return [np.array(embedding.embedding) for embedding in response.data]\n",
    "\n",
    "# Generate embeddings for each chunk size\n",
    "# Iterate over each chunk size and its corresponding chunks in the text_chunks_dict\n",
    "chunk_embeddings_dict = {size: create_embeddings(chunks) for size, chunks in tqdm(text_chunks_dict.items(), desc=\"Generating Embeddings\")}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performing Semantic Search\n",
    "We use cosine similarity to find the most relevant text chunks for a user query."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cosine_similarity(vec1, vec2):\n",
    "    \"\"\"\n",
    "    Computes cosine similarity between two vectors.\n",
    "\n",
    "    Args:\n",
    "    vec1 (np.ndarray): First vector.\n",
    "    vec2 (np.ndarray): Second vector.\n",
    "\n",
    "    Returns:\n",
    "    float: Cosine similarity score.\n",
    "    \"\"\"\n",
    "\n",
    "    # Compute the dot product of the two vectors\n",
    "    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):\n",
    "    \"\"\"\n",
    "    Retrieves the top-k most relevant text chunks.\n",
    "    \n",
    "    Args:\n",
    "    query (str): User query.\n",
    "    text_chunks (List[str]): List of text chunks.\n",
    "    chunk_embeddings (List[np.ndarray]): Embeddings of text chunks.\n",
    "    k (int): Number of top chunks to return.\n",
    "    \n",
    "    Returns:\n",
    "    List[str]: Most relevant text chunks.\n",
    "    \"\"\"\n",
    "    # Generate an embedding for the query - pass query as a list and get first item\n",
    "    query_embedding = create_embeddings([query])[0]\n",
    "    \n",
    "    # Calculate cosine similarity between the query embedding and each chunk embedding\n",
    "    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]\n",
    "    \n",
    "    # Get the indices of the top-k most similar chunks\n",
    "    top_indices = np.argsort(similarities)[-k:][::-1]\n",
    "    \n",
    "    # Return the top-k most relevant text chunks\n",
    "    return [text_chunks[i] for i in top_indices]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['AI enables personalized medicine by analyzing individual patient data, predicting treatment \\nresponses, and tailoring interventions. Personalized medicine enhances treatment effectiveness \\nand reduces adverse effects. \\nRobotic Surgery \\nAI-powered robotic s', ' analyzing biological data, predicting drug \\nefficacy, and identifying potential drug candidates. AI-powered systems reduce the time and cost \\nof bringing new treatments to market. \\nPersonalized Medicine \\nAI enables personalized medicine by analyzing indiv', 'g \\npatient outcomes, and assisting in treatment planning. AI-powered tools enhance accuracy, \\nefficiency, and patient care. \\nDrug Discovery and Development \\nAI accelerates drug discovery and development by analyzing biological data, predicting drug \\neffica', 'mains. \\nThese applications include: \\nHealthcare \\nAI is transforming healthcare through applications such as medical diagnosis, drug discovery, \\npersonalized medicine, and robotic surgery. AI-powered tools can analyze medical images, \\npredict patient outcom', 'Personalized Learning \\nAI enables personalized learning experiences by adapting to individual student needs and \\nlearning styles. AI-powered platforms provide customized content, feedback, and pacing, \\nenhancing student engagement and outcomes. \\nAdaptive A']\n"
     ]
    }
   ],
   "source": [
    "# Load the validation data from a JSON file\n",
    "with open('data/val.json') as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "# Extract the first query from the validation data\n",
    "query = data[3]['question']\n",
    "\n",
    "# Retrieve relevant chunks for each chunk size\n",
    "retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}\n",
    "\n",
    "# Print retrieved chunks for chunk size 256\n",
    "print(retrieved_chunks_dict[256])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generating a Response Based on Retrieved Chunks\n",
    "Let's  generate a response based on the retrieved text for chunk size `256`."
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
      "AI contributes to personalized medicine by analyzing individual patient data, predicting treatment responses, and tailoring interventions. This enables personalized medicine to enhance treatment effectiveness and reduce adverse effects.\n"
     ]
    }
   ],
   "source": [
    "# Define the system prompt for the AI assistant\n",
    "system_prompt = \"You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'\"\n",
    "\n",
    "def generate_response(query, system_prompt, retrieved_chunks, model=\"meta-llama/Llama-3.2-3B-Instruct\"):\n",
    "    \"\"\"\n",
    "    Generates an AI response based on retrieved chunks.\n",
    "\n",
    "    Args:\n",
    "    query (str): User query.\n",
    "    retrieved_chunks (List[str]): List of retrieved text chunks.\n",
    "    model (str): AI model.\n",
    "\n",
    "    Returns:\n",
    "    str: AI-generated response.\n",
    "    \"\"\"\n",
    "    # Combine retrieved chunks into a single context string\n",
    "    context = \"\\n\".join([f\"Context {i+1}:\\n{chunk}\" for i, chunk in enumerate(retrieved_chunks)])\n",
    "    \n",
    "    # Create the user prompt by combining the context and the query\n",
    "    user_prompt = f\"{context}\\n\\nQuestion: {query}\"\n",
    "\n",
    "    # Generate the AI response using the specified model\n",
    "    response = client.chat.completions.create(\n",
    "        model=model,\n",
    "        temperature=0,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": system_prompt},\n",
    "            {\"role\": \"user\", \"content\": user_prompt}\n",
    "        ]\n",
    "    )\n",
    "\n",
    "    # Return the content of the AI response\n",
    "    return response.choices[0].message.content\n",
    "\n",
    "# Generate AI responses for each chunk size\n",
    "ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}\n",
    "\n",
    "# Print the response for chunk size 256\n",
    "print(ai_responses_dict[256])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluating the AI Response\n",
    "We score responses based on faithfulness and relevancy using powerfull llm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define evaluation scoring system constants\n",
    "SCORE_FULL = 1.0     # Complete match or fully satisfactory\n",
    "SCORE_PARTIAL = 0.5  # Partial match or somewhat satisfactory\n",
    "SCORE_NONE = 0.0     # No match or unsatisfactory"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define strict evaluation prompt templates\n",
    "FAITHFULNESS_PROMPT_TEMPLATE = \"\"\"\n",
    "Evaluate the faithfulness of the AI response compared to the true answer.\n",
    "User Query: {question}\n",
    "AI Response: {response}\n",
    "True Answer: {true_answer}\n",
    "\n",
    "Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.\n",
    "\n",
    "INSTRUCTIONS:\n",
    "- Score STRICTLY using only these values:\n",
    "    * {full} = Completely faithful, no contradictions with true answer\n",
    "    * {partial} = Partially faithful, minor contradictions\n",
    "    * {none} = Not faithful, major contradictions or hallucinations\n",
    "- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "RELEVANCY_PROMPT_TEMPLATE = \"\"\"\n",
    "Evaluate the relevancy of the AI response to the user query.\n",
    "User Query: {question}\n",
    "AI Response: {response}\n",
    "\n",
    "Relevancy measures how well the response addresses the user's question.\n",
    "\n",
    "INSTRUCTIONS:\n",
    "- Score STRICTLY using only these values:\n",
    "    * {full} = Completely relevant, directly addresses the query\n",
    "    * {partial} = Partially relevant, addresses some aspects\n",
    "    * {none} = Not relevant, fails to address the query\n",
    "- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Faithfulness Score (Chunk Size 256): 0.5\n",
      "Relevancy Score (Chunk Size 256): 0.5\n",
      "\n",
      "\n",
      "Faithfulness Score (Chunk Size 128): 0.5\n",
      "Relevancy Score (Chunk Size 128): 0.5\n"
     ]
    }
   ],
   "source": [
    "def evaluate_response(question, response, true_answer):\n",
    "        \"\"\"\n",
    "        Evaluates the quality of an AI-generated response based on faithfulness and relevancy.\n",
    "\n",
    "        Args:\n",
    "        question (str): The user's original question.\n",
    "        response (str): The AI-generated response being evaluated.\n",
    "        true_answer (str): The correct answer used as ground truth.\n",
    "\n",
    "        Returns:\n",
    "        Tuple[float, float]: A tuple containing (faithfulness_score, relevancy_score).\n",
    "                                                Each score is one of: 1.0 (full), 0.5 (partial), or 0.0 (none).\n",
    "        \"\"\"\n",
    "        # Format the evaluation prompts\n",
    "        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(\n",
    "                question=question, \n",
    "                response=response, \n",
    "                true_answer=true_answer,\n",
    "                full=SCORE_FULL,\n",
    "                partial=SCORE_PARTIAL,\n",
    "                none=SCORE_NONE\n",
    "        )\n",
    "        \n",
    "        relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(\n",
    "                question=question, \n",
    "                response=response,\n",
    "                full=SCORE_FULL,\n",
    "                partial=SCORE_PARTIAL,\n",
    "                none=SCORE_NONE\n",
    "        )\n",
    "\n",
    "        # Request faithfulness evaluation from the model\n",
    "        faithfulness_response = client.chat.completions.create(\n",
    "               model=\"meta-llama/Llama-3.2-3B-Instruct\",\n",
    "                temperature=0,\n",
    "                messages=[\n",
    "                        {\"role\": \"system\", \"content\": \"You are an objective evaluator. Return ONLY the numerical score.\"},\n",
    "                        {\"role\": \"user\", \"content\": faithfulness_prompt}\n",
    "                ]\n",
    "        )\n",
    "        \n",
    "        # Request relevancy evaluation from the model\n",
    "        relevancy_response = client.chat.completions.create(\n",
    "                model=\"meta-llama/Llama-3.2-3B-Instruct\",\n",
    "                temperature=0,\n",
    "                messages=[\n",
    "                        {\"role\": \"system\", \"content\": \"You are an objective evaluator. Return ONLY the numerical score.\"},\n",
    "                        {\"role\": \"user\", \"content\": relevancy_prompt}\n",
    "                ]\n",
    "        )\n",
    "        \n",
    "        # Extract scores and handle potential parsing errors\n",
    "        try:\n",
    "                faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())\n",
    "        except ValueError:\n",
    "                print(\"Warning: Could not parse faithfulness score, defaulting to 0\")\n",
    "                faithfulness_score = 0.0\n",
    "                \n",
    "        try:\n",
    "                relevancy_score = float(relevancy_response.choices[0].message.content.strip())\n",
    "        except ValueError:\n",
    "                print(\"Warning: Could not parse relevancy score, defaulting to 0\")\n",
    "                relevancy_score = 0.0\n",
    "\n",
    "        return faithfulness_score, relevancy_score\n",
    "\n",
    "# True answer for the first validation data\n",
    "true_answer = data[3]['ideal_answer']\n",
    "\n",
    "# Evaluate response for chunk size 256 and 128\n",
    "faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)\n",
    "faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)\n",
    "\n",
    "# print the evaluation scores\n",
    "print(f\"Faithfulness Score (Chunk Size 256): {faithfulness}\")\n",
    "print(f\"Relevancy Score (Chunk Size 256): {relevancy}\")\n",
    "\n",
    "print(f\"\\n\")\n",
    "\n",
    "print(f\"Faithfulness Score (Chunk Size 128): {faithfulness2}\")\n",
    "print(f\"Relevancy Score (Chunk Size 128): {relevancy2}\")"
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
