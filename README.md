# Biomedical Abstract Simplification Using Large Language Models (LLMs)

## 📌 Project Overview

Biomedical research papers contain complex medical terminology and technical sentences that can be difficult for patients, students, researchers, and non-medical users to understand.

This project presents a **Biomedical Abstract Simplification System** that uses a pretrained **FLAN-T5 transformer model** to convert complex biomedical and medical text into simpler and more understandable language.

The application provides three simplification levels:

- **Mild** – Provides a short and concise explanation.
- **Medium** – Provides a simple explanation with additional context.
- **Strong** – Provides a more detailed explanation using easy-to-understand language.

The system is implemented as an interactive **Streamlit web application** with authentication, history management, dashboard visualization, AI scoring, and PDF report generation.


## 🎯 Objectives

The major objectives of this project are:

1. To develop an AI-based biomedical text simplification system.
2. To convert complex medical terminology into understandable language.
3. To provide multiple levels of simplification according to user requirements.
4. To develop an easy-to-use web interface using Streamlit.
5. To maintain the history of previously simplified text.
6. To provide an AI-based score for generated simplification.
7. To allow users to download simplification results as PDF reports.
8. To provide secure user authentication using password hashing.



## 🚀 Features

### 1. Biomedical Text Simplification

Users can enter a medical term, medicine name, or biomedical text and generate a simplified explanation using the FLAN-T5 model.

### 2. Multiple Simplification Levels

The application provides three levels:

| Level | Description |
|---|---|
| Mild | Generates a short and concise medical explanation. |
| Medium | Provides a clearer explanation with more context. |
| Strong | Generates a more detailed explanation using simpler language. |

### 3. User Authentication

The application provides:

- User registration
- Login
- Password verification
- Password reset
- Logout
- Remember-login functionality

Passwords are stored using **bcrypt hashing** rather than plain text.

### 4. Simplification History

The system stores previous simplification results so that users can review their previous inputs and generated outputs.

### 5. Dashboard

The dashboard provides a summary of system usage, including the total number of simplifications and simplification-level statistics.

### 6. AI Score

The application provides an AI-based score associated with the generated simplification to give the user an indication of the output quality.

> Note: The score is an application-level heuristic and should not be interpreted as a clinical accuracy or medical confidence measurement.

### 7. PDF Report Generation

Users can generate and download a PDF containing the original input and simplified output.

### 8. Background and User Interface

The application provides a customized Streamlit interface with a background image and organized navigation.



## 🧠 Model Used

### FLAN-T5

The project uses **FLAN-T5**, an instruction-tuned version of the T5 transformer architecture.

FLAN-T5 is suitable for this project because it can follow natural-language instructions and perform text-generation tasks.

The application uses the model to generate explanations based on the selected simplification level.

### Model Workflow

```text
User Input
     ↓
Input Validation
     ↓
Simplification Level Selection
     ↓
Prompt Generation
     ↓
FLAN-T5 Tokenization
     ↓
FLAN-T5 Model
     ↓
Text Generation
     ↓
Simplified Output
     ↓
AI Score + History + PDF
