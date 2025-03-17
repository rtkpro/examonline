import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
import json
import requests
import streamlit.components.v1 as components

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")

def extract_qa(llm_response):
    try:
        if not llm_response or not llm_response.content.strip():
            return None
        json_string = llm_response.content.strip()
        try:
            qa_list = json.loads(json_string)
            return qa_list
        except json.JSONDecodeError:
            try:
                json_string = json_string.replace("\n", "").replace("```json", "").replace("```", "")
                qa_list = json.loads(json_string)
                return qa_list
            except json.JSONDecodeError:
                try:
                    json_string = json_string.replace(",]", "]").replace(",}", "}")
                    qa_list = json.loads(json_string)
                    return qa_list
                except json.JSONDecodeError:
                    return None
    except Exception as e:
        st.write(f"Error during JSON extraction: {e}")
        return None

def question_Generate(keyword, experience):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    mcq_prompt = [("system", f"""You are an expert MCQ (Multiple Choice Question) generator for coding-related topics. Your task is to create five (5) high-quality MCQ questions based on the user's provided programming language and experience.

        **Constraints:**
        * Each question must have exactly four (4) distinct options (A, B, C, D).
        * One and only one option must be the correct answer.
        * The questions should be relevant to the programming language: '{keyword}' and experience level: '{experience}'.
        * The questions should focus on code snippets, outputs, or concepts related to the given language.
        * The questions should be challenging but fair, suitable for someone with the specified experience.
        * Avoid overly complex or ambiguous questions.
        * Format your response strictly as a JSON list of dictionaries.
        * Each dictionary must have the following keys: "question", "options" (a list of four strings), and "answer" (the correct answer string).
        * Ensure the JSON is valid and parsable.
        * Do not include any introductory or explanatory text outside of the JSON.
        * Do not include any markdown code blocks, just the raw json.

        **Example JSON Output Format:**
        [
            {{
                "question": "What is the output of the following Python code: print(2 + 2)?",
                "options": ["3", "4", "5", "6"],
                "answer": "4"
            }},
            {{
                "question": "Which keyword is used to define a function in JavaScript?",
                "options": ["def", "function", "define", "method"],
                "answer": "function"
            }},
            // ... 3 more questions ...
        ]

        **User Programming Language and Experience:**
        Language: {keyword}
        Experience: {experience}
        """) ,
        ("human", "")
        ]

    code_prompt = f"""You are an expert programming *coding question generator*. Your task is to create two (2) coding questions tailored for a programmer with {experience} level experience in {keyword}.

        **Constraints:**
        * Generate exactly 2 coding questions.
        * Each question should assess the programmer's understanding and practical skills in {keyword}.
        * Format your response strictly as a JSON list of dictionaries.
        * Each dictionary must contain the keys "question".
        * Ensure the JSON is valid and parsable.
        * Do not include any introductory or explanatory text outside of the JSON.
        * Do not include any markdown code blocks, just the raw JSON.

        **Example JSON Output Format:**
        [
            {{
                "question": "Write the code of [Specific {keyword} concept] and provide an example.",
            }},
            {{
                "question": "Create a function to perform task [Another specific {keyword} concept]",
            }},
        ]

        **Programming Skillset and Experience:**
        Skillset: {keyword}
        Experience: {experience}

        Generate the coding questions."""

    sub_prompt = f"""You are an expert programming *theoretical question generator*. Your task is to create two (2) subjective theoretical programming questions tailored for a programmer with {experience} level experience in {keyword}.

        **Constraints:**
        * Generate exactly 2 subjective theoretical programming questions.
        * Each question should be designed to assess the programmer's understanding and practical skills in {keyword}.
        * Format your response strictly as a JSON list of dictionaries.
        * Each dictionary must contain the keys "question".
        * Ensure the JSON is valid and parsable.
        * Do not include any introductory or explanatory text outside of the JSON.
        * Do not include any markdown code blocks, just the raw json.

        **Example JSON Output Format:**
        [
            {{
                "question": "Explain the concept of [Specific {keyword} concept] and provide an example.",
            }},
            {{
                "question": "Describe a scenario where you would use [Another specific {keyword} concept] and why.",
            }},
        ]

        **Programming Skillset and Experience:**
        Skillset: {keyword}
        Experience: {experience}

        Generate the programming questions."""

    mcq_response = llm.invoke(mcq_prompt)
    code_response = llm.invoke(code_prompt)
    sub_response = llm.invoke(sub_prompt)

    mcq_data = extract_qa(mcq_response)
    code_data = extract_qa(code_response)
    sub_data = extract_qa(sub_response)

    return mcq_data, code_data, sub_data

def evaluate_answer(question, student_answer):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    if not student_answer.strip():
        return {"score": 0, "feedback": "Answer was blank or empty.", "result": "fail"}

    prompt = f"""You are an expert exam evaluator. Your task is to accurately evaluate the student's answer {student_answer} for the given question {question['question']} accordingly.

        **Question:** {question['question']}
        **Student Answer:** {student_answer}

        **Evaluation Criteria:**
        1. Assess the correctness and completeness of the student's answer.
        2. Provide a numerical score as a percentage (0% to 100%), where 100% represents a perfect answer.
        3. Provide detailed feedback explaining the score and highlighting areas of strength and weakness.
        4. If the score percentage is greater than 60% consider the result as "pass" else "fail".
        5. Handle both "coding" and "subjective/theoretical" questions appropriately.

        **Constraints:**
        * Provide a JSON response with the keys "score" (integer), "feedback" (string), and "result" (string - "pass" or "fail").
        * Ensure the JSON is valid and parsable.
        * Do not include any introductory or explanatory text outside of the JSON.

        **Example JSON Output Format:**
            1. {{
                "score": 85,
                "feedback": "...",
                "result": "pass"
            }}

            2. {{
                "score": 0,
                "feedback": "...",
                "result": "fail"
            }}

        Evaluate the student's answer and provide the JSON response."""

    response = llm.invoke(prompt)
    try:
        json_string = response.content.strip().replace('```json', '').replace('```', '').strip()
        evaluation = json.loads(json_string)
        return evaluation
    except json.JSONDecodeError:
        st.error("JSON Decode Error: Could not parse LLM evaluation response as JSON.")
        print("Raw evaluation response:", response.content)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def submit_test_results(mcq_score, subjective_evaluations, code_evaluations, email, test_id):
    """Submits the test results to the API."""

    subjective_score = sum(eval.get('score', 0) for eval in subjective_evaluations.values()) /len(subjective_evaluations) if subjective_evaluations else 0
    coding_score= sum(eval.get('score', 0) for eval in code_evaluations.values()) / len(code_evaluations) if code_evaluations else 0

    try:
        response = requests.get(f"https://doskr.com/RESTAPI/udpatescore.php?email={email}&test_id={test_id}&mcq_score={mcq_score}&subjective_score={subjective_score}&coding_score={coding_score}")
        st.write(response)
        response.raise_for_status()
        st.success("Test results submitted successfully!")

    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting test results: {e}")


def camera_app():
    camera_html = """
    <script>
    var video = document.createElement('video');
    video.setAttribute('autoplay', '');
    video.setAttribute('muted', '');
    video.style.width = '100%';
    video.style.height = 'auto';

    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        video.srcObject = stream;
        document.body.appendChild(video); // Append to body so streamlit can see it.
    })
    .catch(function(error) {
        console.error('Error accessing camera:', error);
    });
    </script>
    """

    components.html(camera_html, height=480)

# if __name__ == "__main__":
#     camera_app()

st.title("AI-Powered MCQ, Subjective, and Coding Test")
st.write("Click on Start Audio/Video recording first")

keywords = "Python"
experience = "2 years"
query_params = st.query_params
if "keywords" in query_params:
    keywords = query_params["keywords"]
if "experience" in query_params:
    try:
        experience = int(query_params["experience"])
        experience = str(experience) + " years "
    except ValueError:
        st.error("Invalid experience value in URL. Must be an integer.")
        st.stop()

if 'exam_started' not in st.session_state:
    st.session_state.exam_started = False
if 'mcq_questions' not in st.session_state:
    st.session_state.mcq_questions = None
if 'subjective_questions' not in st.session_state:
    st.session_state.subjective_questions = None
if 'code_questions' not in st.session_state:
    st.session_state.code_questions = None
if 'mcq_answers' not in st.session_state:
    st.session_state.mcq_answers = {}
if 'subjective_answers' not in st.session_state:
    st.session_state.subjective_answers = {}
if 'code_answers' not in st.session_state:
    st.session_state.code_answers = {}
if 'mcq_evaluations' not in st.session_state:
    st.session_state.mcq_evaluations = {}
if 'subjective_evaluations' not in st.session_state:
    st.session_state.subjective_evaluations = {}
if 'code_evaluations' not in st.session_state:
    st.session_state.code_evaluations = {}
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'video_started' not in st.session_state:
    st.session_state.video_started = False  # Start video by default
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False # Track if questions have been generated.

col1, col2 = st.columns([3, 1])

with col2:
    if not st.session_state.video_started:
        if st.button("Start Camera"):
            try:
                camera_app()
                st.session_state.video_started = True
                st.rerun()
            except Exception as e:
                st.error(f"Error starting camera: {e}")
    else:
        camera_app()

with col1:
    if not st.session_state.exam_started and not st.session_state.evaluation_done:
        st.write(f"Current keyword: {keywords}, Experience level: {experience}")
        if st.button("Start All Tests"):
            st.session_state.exam_started = True
            st.session_state.mcq_questions, st.session_state.code_questions, st.session_state.subjective_questions = question_Generate(keywords, experience)
            if not st.session_state.mcq_questions or not st.session_state.subjective_questions or not st.session_state.code_questions:
                st.error("Failed to generate questions. Please try again.")
                st.session_state.exam_started = False
                st.stop()
            st.session_state.questions_generated = True # Questions are now generated.
            st.rerun() # Refresh to show questions immediately.

    elif st.session_state.exam_started and st.session_state.questions_generated:
        st.subheader("MCQ Questions")
        if st.session_state.mcq_questions:
            for i, q in enumerate(st.session_state.mcq_questions):
                if isinstance(q, dict):
                    st.subheader(q.get('question', 'Question not found'))
                    selected_index = None
                    if str(i) in st.session_state.mcq_answers and st.session_state.mcq_answers[str(i)] in q.get('options', []):
                        selected_index = q['options'].index(st.session_state.mcq_answers[str(i)])

                    answer = st.radio(
                        "Choose one option",
                        q.get('options', []),
                        index=selected_index,
                        key=f"mcq_{q.get('question', f'question_{i}')}_{i}"
                    )
                    st.session_state.mcq_answers[str(i)] = answer
                else:
                    st.write(f"Invalid question format at index {i}: {q}")

        st.subheader("Subjective Questions")
        if st.session_state.subjective_questions:
            for i, question in enumerate(st.session_state.subjective_questions):
                st.subheader(f"Question {i + 1}:")
                st.write(question["question"])
                student_answer = st.text_area(f"Your Answer (Question {i + 1})",
                                                value=st.session_state.subjective_answers.get(str(i), ""),
                                                key=f"subjective_answer_{i}")
                st.session_state.subjective_answers[str(i)] = student_answer

        st.subheader("Coding Questions")
        if st.session_state.code_questions:
            for i, question in enumerate(st.session_state.code_questions):
                st.subheader(f"Coding Question {i + 1}:")
                st.write(question["question"])
                student_code = st.text_area(f"Your Code (Question {i + 1})",
                                             value=st.session_state.code_answers.get(str(i), ""),
                                             key=f"code_answer_{i}")
                st.session_state.code_answers[str(i)] = student_code

        if st.button("Submit All Tests"):
            if st.session_state.mcq_questions:
                if None in st.session_state.mcq_answers.values():
                    st.warning("Please answer all MCQ questions before submitting.")
                    st.stop()

            if st.session_state.subjective_questions:
                for i, question in enumerate(st.session_state.subjective_questions):
                    evaluation = evaluate_answer(question, st.session_state.subjective_answers.get(str(i), ""))
                    st.session_state.subjective_evaluations[str(i)] = evaluation

            if st.session_state.code_questions:
                for i, question in enumerate(st.session_state.code_questions):
                    evaluation = evaluate_answer(question, st.session_state.code_answers.get(str(i), ""))
                    st.session_state.code_evaluations[str(i)] = evaluation

            st.session_state.exam_started = False
            st.session_state.evaluation_done = True
            st.session_state.video_started = False  # Stop video stream
            st.rerun()

    elif st.session_state.evaluation_done:
        st.subheader("MCQ Evaluations:")
        if st.session_state.mcq_questions:
            def calculate_score(student_answers):
                correct_answers = 0
                for i, answer in enumerate(student_answers.values()):
                    if answer is not None and i < len(st.session_state.mcq_questions):
                        if answer == st.session_state.mcq_questions[i]['answer']:
                            correct_answers += 1
                return correct_answers
            for i, question in enumerate(st.session_state.mcq_questions):
                st.subheader(f"MCQ Evaluation for Question {i + 1}:")
                st.write(f"Question: {question['question']}")
                st.write(f"Correct Answer: {question['answer']}")
                st.write(f"Your Answer: {st.session_state.mcq_answers.get(str(i), 'Not Answered')}")
                st.write("-" * 20)
            score = calculate_score(st.session_state.mcq_answers)
            total_questions = len(st.session_state.mcq_questions)
            percentage = (score / total_questions) * 100
            st.write(f"Your MCQ score is: {score}/{total_questions} ({percentage:.2f}%)")
            st.write("-" * 20)

        st.subheader("Subjective Evaluations:")
        if st.session_state.subjective_questions:
            for i, question in enumerate(st.session_state.subjective_questions):
                st.subheader(f"Subjective Evaluation for Question {i + 1}:")
                st.write(f"Question: {question['question']}")
                evaluation = st.session_state.subjective_evaluations.get(str(i))
                if evaluation and 'score' in evaluation and 'feedback' in evaluation:
                    st.write(f"Score: {evaluation['score']}%")
                    st.write(f"Feedback: {evaluation['feedback']}")
                else:
                    st.write("Evaluation not available or invalid format.")
                st.write("-" * 20)

        st.subheader("Coding Evaluations:")
        if st.session_state.code_questions:
            for i, question in enumerate(st.session_state.code_questions):
                st.subheader(f"Coding Evaluation for Question {i + 1}:")
                st.write(f"Question: {question['question']}")
                evaluation = st.session_state.code_evaluations.get(str(i))
                if evaluation and 'score' in evaluation and 'feedback' in evaluation:
                    st.write(f"Score: {evaluation['score']}%")
                    st.write(f"Feedback: {evaluation['feedback']}")
                else:
                    st.write("Evaluation not available or invalid format.")
                st.write("-" * 20)

        email = query_params.get("email", None)
        test_id = query_params.get("test_id", None)

        if email and test_id:
            submit_test_results(percentage,
                                 st.session_state.subjective_evaluations,
                                 st.session_state.code_evaluations,
                                 email,
                                 test_id)
        else:
            st.warning("Email and Test ID must be provided in the URL to submit results.")
