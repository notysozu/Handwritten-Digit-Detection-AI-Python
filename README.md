# Handwritten Digit Recognition using MNIST + Gemma (Ollama)

## Overview

This project is a **basic academic assignment** implemented as part of official coursework.  
The objective is to demonstrate the integration of:

- A **pretrained MNIST handwritten digit recognition model** (for prediction)
- The **Gemma language model via Ollama** (for explanation only)

The project is fully offline and runs locally on the system.

---

## Assignment Objective

- Use a **pretrained MNIST model** to recognize handwritten digits
- Integrate the **Gemma model using Ollama**
- Ensure that **Gemma is used only for explanation**, not for prediction
- Maintain a **simple, beginner-friendly Python project structure**

---

## Important Constraints (As per Assignment)

- ❌ No web UI
- ❌ No cloud APIs
- ❌ No online LLMs
- ❌ Gemma must NOT predict digits
- ✅ Fully offline execution
- ✅ Clean separation between ML prediction and LLM explanation

---

## Project Workflow

1. A handwritten digit image is loaded from disk
2. A pretrained MNIST CNN predicts:
   - The digit (0–9)
   - The confidence score
3. The prediction result (digit + confidence) is sent to **Gemma via Ollama**
4. Gemma returns a **natural language explanation** of the confidence
5. Results are printed in the terminal

---

## Project Structure

handwritten-digit-recognition-in-python/
│
├── data/
│ └── sample_digits/ # Input handwritten digit images
│
├── model/
│ └── mnist_model.py # MNIST CNN definition and inference logic
│
├── inference/
│ └── predict.py # Image preprocessing and prediction
│
├── ollama/
│ └── gemma.py # Gemma explanation via Ollama API
│
├── main.py # Main execution script
├── requirements.txt
└── README.md


---

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **Ollama**
- **Gemma Language Model**

---

## Setup Instructions

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure Ollama is running

```bash
ollama serve
```
### 4. Pull Gemma model

```bash
ollama pull gemma:2b
```

## Running the Project

1. Place a handwritten digit image inside:
```bash
data/sample_digits/
```

2. Run the project:
```bash
python main.py
```