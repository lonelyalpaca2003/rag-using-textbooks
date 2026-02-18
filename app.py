import gradio as gr
from src.indexing import create_query_engine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters 
from src.prompts import QUIZ_GENERATION, SUMMARIZE_LECTURE, FIND_TEXTBOOK_PAGES, EXAM_PREP
from dotenv import load_dotenv

load_dotenv()

query_engine, index = create_query_engine()

def ask_question(prompt):
    response =  query_engine.query(prompt)
    return response.response

def generate_quiz(lecture_num, num_questions):
    prompt = QUIZ_GENERATION.format(source = f" ST443_Lecture {lecture_num}", num_questions = num_questions)
    response = query_engine.query(prompt)
    return response.response

def summarize_lecture(lecture_num):
    filters = MetadataFilters(filters = [
        MetadataFilter(key = "file_name", 
                       value = f"ST443_Lecture_{lecture_num}.pdf", 
                       operator = "==")
                       ])
    filtered_qe = index.as_query_engine(similarity_top_k = 50, 
                                        filters = filters, 
                                        response_mode = "tree_summarize")
    prompt = SUMMARIZE_LECTURE.format(lecture = f" ST443_Lecture_{lecture_num}")
    response = filtered_qe.query(prompt)
    
    return response.response

def find_textbook_pages(textbook, topic):
    prompt = FIND_TEXTBOOK_PAGES.format(textbook = textbook, topic = topic)
    response = query_engine.query(prompt)
    return response.response

def exam_prep(topic):
    prompt = EXAM_PREP.format( topic = topic)
    response = query_engine.query(prompt)
    return response.response

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ML Study Assistant")
    gr.Markdown("Ask questions about your Machine Learning course materials")
    
    with gr.Tab("Ask Questions"):
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is regularisation?",
                    lines=3
                )
                ask_btn = gr.Button("Ask", variant="primary")
                
                gr.Examples(
                    examples=[
                        "What is gradient descent?",
                        "Explain the bias-variance tradeoff",
                        "What is the difference between ridge and lasso regression?"
                    ],
                    inputs=question_input
                )
            
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
        
        ask_btn.click(ask_question, inputs=question_input, outputs=answer_output)
    
    with gr.Tab(" Generate Quiz"):
        with gr.Row():
            with gr.Column():
                quiz_topic = gr.Textbox(
                    label="Topic",
                    placeholder="regularisation"
                )
                num_q = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Questions"
                )
                quiz_btn = gr.Button("Generate Quiz", variant="primary")
            
            with gr.Column():
                quiz_output = gr.Markdown(label="Quiz Questions")
        
        quiz_btn.click(generate_quiz, inputs=[quiz_topic, num_q], outputs=quiz_output)
    
    with gr.Tab(" Summarize Lecture"):
        with gr.Row():
            with gr.Column():
                lecture_num = gr.Number(
                    label="Lecture Number",
                    value=5,
                    precision=0
                )
                summarize_btn = gr.Button("Summarize", variant="primary")
            
            with gr.Column():
                summary_output = gr.Markdown(label="Lecture Summary")
        
        summarize_btn.click(summarize_lecture, inputs=lecture_num, outputs=summary_output)
    
    with gr.Tab("Find in Textbook"):
        with gr.Row():
            with gr.Column():
                topic_input = gr.Textbox(
                    label="Topic to Find",
                    placeholder="ridge regression"
                )
                textbook_input = gr.Textbox(label = "Textbook to search through", placeholder = "ISLR")

                find_btn = gr.Button("Find Pages", variant="primary")
            
            with gr.Column():
                pages_output = gr.Markdown(label="Textbook Pages")
        
        find_btn.click(find_textbook_pages, inputs=[topic_input, textbook_input], outputs=pages_output)
    
    with gr.Tab(" About"):
        gr.Markdown("""
        ## How to Use
        
        **Ask Questions**: Get answers with citations from your course materials
        
        **Generate Quiz**: Create practice questions on specific topics
        
        **Summarize Lecture**: Get organized summaries of lecture content
        
        **Find in Textbook**: Locate topics in ISLR textbook
        
        ## Indexed Materials
        - ST443 Lecture Slides
        - ISLR Textbook (Introduction to Statistical Learning)
        - Elements of Statistical Learning
        
        ## Tips
        - Be specific in your questions for better results
        - Check the sources to verify information
        - Use quiz generation to test your understanding
        """)

if __name__ == "__main__":
    demo.launch(share=False)



