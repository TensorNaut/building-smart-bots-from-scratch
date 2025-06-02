# 🧠 Building Smart Bots from Scratch

Welcome to **Smart Bots from Scratch**, a project that takes you on a journey to build intelligent chatbot systems **step-by-step**, starting from the basics and gradually evolving into highly intelligent AI-powered assistants.

---

## 🚀 Project Overview

This project is structured in **progressive levels**, each introducing more intelligence and advanced techniques:

### ✅ Level 1: Rule-Based Chatbot (Current)

- TF-IDF Vectorization of dataset questions.
- Cosine Similarity for matching user input with dataset.
- Rule-based response selection using a custom QnA dataset.
- Deployed via **Streamlit** for easy interaction.

📁 Folder: `level-1-rule-based/`  
🧠 Main Script: `chatbot.py`  
📊 Dataset: `data/conversation.csv`  
🧹 Cleaned Dataset: `data/cleaned_conversation.csv`  
🌐 Streamlit App: `level-1-rule-based/streamlit_app.py`

---

## 🔧 How It Works (Level 1)

1. Load the cleaned QnA dataset.  
2. Convert dataset questions and user input to **TF-IDF vectors**.  
3. Calculate **cosine similarity** to find the best match.  
4. Respond with the matching answer.  
5. Repeat until the user exits.  

> This forms the foundation for upcoming AI-powered chatbot levels.

---

## 🛠️ How to Run

#### ▶️ Run from Command Line
```bash
pip install pandas scikit-learn
python level-1-rule-based/chatbot.py
```

#### 🖥️ Run Streamlit App
```bash
pip install streamlit pandas scikit-learn
streamlit run level-1-rule-based/streamlit_app.py
```

---

🌱 **What’s Coming Next?**

🔹 **Level 2: Embeddings + FAISS (Coming Soon)**  
- Use pre-trained sentence embeddings (e.g., SBERT) to encode questions and user input.  
- Index embeddings using FAISS for efficient nearest-neighbor search.  
- Retrieve and return multiple possible answers ranked by similarity.

🔹 **Level 3: RAG with Transformers (Planned)**  
- Build a Retrieval-Augmented Generation pipeline using LangChain + OpenAI/GPT.  
- Retrieve top-k QnA pairs from a vector store.  
- Pass retrieved context to an LLM to generate fluent and context-aware responses.  
- Support multi-turn dialogue with conversational memory.

🔹 **Level 4: ...................... (Future Surprise)**  
Advanced integrations and real-world intelligence... Stay tuned! 👀

---

💡 **Why This Project?**  
This is more than just a chatbot repo — it's a learning path.  
Whether you're just starting out or aiming to build cutting-edge conversational agents, this repo is crafted to teach you the **how** and **why** behind each method.

---

🤝 **Contributions**  
Ideas, feedback, improvements? You're welcome!  
Feel free to fork, PR, or raise an issue.

---

✨ **Stay Tuned**  
🚧 Level 2 in progress...  
📚 More intelligent bots are coming soon!  
🎯 Step-by-step. No shortcuts. Only smart moves.

---

💬 **Happy Chatbot Building!**