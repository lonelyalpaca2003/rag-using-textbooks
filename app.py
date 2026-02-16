import gradio as gr
from src.indexing import create_query_engine
from src.prompts import QUIZ_GENERATION, SUMMARIZE_LECTURE, FIND_TEXTBOOK_PAGES, EXAM_PREP
from dotenv import load_dotenv

load_dotenv()

query_engine = create_query_engine()

def ask_question(prompt):
    response =  query_engine.query(prompt)
    return response.response

def generate_quiz(lecture_num, num_questions):
    prompt = QUIZ_GENERATION.format(source = f"Lecture {lecture_num}", num_questions = num_questions)
    response = query_engine.query(prompt)
    return response.response

def summarize_lecture(lecture_num):
    prompt = SUMMARIZE_LECTURE.format(lecture = f"Lecture {lecture_num}")
    response = query_engine.query(prompt)
    return response.response

def find_textbook_pages(textbook, topic):
    prompt = FIND_TEXTBOOK_PAGES.format(textbook = textbook, topic = topic)
    response = query_engine.query(prompt)
    return response.response

def exam_prep(topic):
    prompt = EXAM_PREP.format( topic = topic)
    response = query_engine.query(prompt)
    return response.response



