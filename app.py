import gradio as gr
from src.indexing import create_query_engine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from src.prompts import QUIZ_GENERATION, SUMMARIZE_LECTURE, FIND_TEXTBOOK_PAGES, EXAM_PREP
from dotenv import load_dotenv
import tempfile

load_dotenv()

query_engine, index = create_query_engine()

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    similarity_top_k=10,
    verbose=False,
)

LECTURE_CHOICES = [str(i) for i in range(1, 11)]

TEXTBOOK_CHOICES = {
    "ISLR": "ISLRv2_corrected_June_2023.pdf",
    "Elements of Statistical Learning": "elements_of_statistical_learning.pdf",
    "Mathematics for Machine Learning": "mml-book.pdf",
}


def ask_question(prompt):
    response = query_engine.query(prompt)
    return response.response


def generate_quiz(lecture_num, num_questions):
    filters = MetadataFilters(filters=[
        MetadataFilter(key="file_name",
                       value=f"ST443_Lecture_{lecture_num}.pdf",
                       operator="==")
    ])
    filtered_qe = index.as_query_engine(
        similarity_top_k=15,
        filters=filters,
        response_mode="tree_summarize"
    )
    prompt = QUIZ_GENERATION.format(source=f"ST443_Lecture_{lecture_num}.pdf", num_questions=num_questions)
    response = filtered_qe.query(prompt)
    return response.response


def generate_quiz_ui(lecture_num, num_questions):
    result = generate_quiz(lecture_num, int(num_questions))
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
        prefix=f"quiz_lecture_{lecture_num}_"
    )
    tmp.write(result)
    tmp.close()
    return result, gr.update(value=tmp.name, visible=True)


def summarize_lecture(lecture_num):
    filters = MetadataFilters(filters=[
        MetadataFilter(key="file_name",
                       value=f"ST443_Lecture_{lecture_num}.pdf",
                       operator="==")
    ])
    filtered_qe = index.as_query_engine(
        similarity_top_k=20,
        filters=filters,
        response_mode="tree_summarize"
    )
    response = filtered_qe.query(SUMMARIZE_LECTURE)
    return response.response


def find_textbook_pages(textbook, topic):
    filename = TEXTBOOK_CHOICES.get(textbook)
    if not filename:
        return f"Textbook '{textbook}' not found in database"

    filters = MetadataFilters(filters=[
        MetadataFilter(key="file_name", value=filename),
        MetadataFilter(key="doc_type", value="textbook")
    ])
    filtered_qe = index.as_query_engine(
        similarity_top_k=15,
        filters=filters
    )
    prompt = FIND_TEXTBOOK_PAGES.format(textbook=textbook, topic=topic)
    response = filtered_qe.query(prompt)

    sources_text = "\n\n**Retrieved pages:**\n"
    for node in response.source_nodes:
        sources_text += f"- Page {node.metadata['page_num']}, Score: {node.score:.2f}\n"

    return response.response + sources_text


def exam_prep(topic):
    prompt = EXAM_PREP.format(topic=topic)
    response = query_engine.query(prompt)
    return response.response


def chat_respond(message, history):
    response = chat_engine.chat(message)
    history = history + [[message, response.response]]
    return "", history


def reset_chat():
    chat_engine.reset()
    return [], ""


with gr.Blocks() as demo:
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

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=450, show_label=True)
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask a follow-up question...",
                label="Message",
                scale=4,
                lines=1
            )
            chat_send_btn = gr.Button("Send", variant="primary", scale=1)
        new_chat_btn = gr.Button("New Conversation", variant="secondary")

        chat_send_btn.click(chat_respond, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat_respond, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        new_chat_btn.click(reset_chat, outputs=[chatbot, chat_input])

    with gr.Tab("Generate Quiz"):
        with gr.Row():
            with gr.Column():
                quiz_lecture_num = gr.Dropdown(
                    label="Lecture Number",
                    choices=LECTURE_CHOICES,
                    value="1"
                )
                num_q = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Questions"
                )
                quiz_btn = gr.Button("Generate Quiz", variant="primary")
                download_btn = gr.DownloadButton("Download Quiz (.txt)", visible=False)
            with gr.Column():
                quiz_output = gr.Markdown(label="Quiz Questions")

        quiz_btn.click(generate_quiz_ui, inputs=[quiz_lecture_num, num_q], outputs=[quiz_output, download_btn])

    with gr.Tab("Summarize Lecture"):
        with gr.Row():
            with gr.Column():
                lecture_num = gr.Dropdown(
                    label="Lecture Number",
                    choices=LECTURE_CHOICES,
                    value="5"
                )
                summarize_btn = gr.Button("Summarize", variant="primary")
            with gr.Column():
                summary_output = gr.Markdown(label="Lecture Summary")

        summarize_btn.click(summarize_lecture, inputs=lecture_num, outputs=summary_output)

    with gr.Tab("Find in Textbook"):
        with gr.Row():
            with gr.Column():
                textbook_input = gr.Dropdown(
                    label="Textbook",
                    choices=list(TEXTBOOK_CHOICES.keys()),
                    value="ISLR"
                )
                topic_input = gr.Textbox(
                    label="Topic to Find",
                    placeholder="ridge regression"
                )
                find_btn = gr.Button("Find Pages", variant="primary")
            with gr.Column():
                pages_output = gr.Markdown(label="Textbook Pages")

        find_btn.click(find_textbook_pages, inputs=[textbook_input, topic_input], outputs=pages_output)

    with gr.Tab("Exam Prep"):
        with gr.Row():
            with gr.Column():
                exam_topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="regularisation",
                    lines=1
                )
                exam_btn = gr.Button("Generate Study Guide", variant="primary")
                gr.Examples(
                    examples=["SVMs", "neural networks", "ridge regression", "cross-validation"],
                    inputs=exam_topic_input
                )
            with gr.Column():
                exam_output = gr.Markdown(label="Study Guide")

        exam_btn.click(exam_prep, inputs=exam_topic_input, outputs=exam_output)

    with gr.Tab("About"):
        gr.Markdown("""
        ## How to Use

        **Ask Questions**: Get answers from your course materials

        **Chat**: Have a multi-turn conversation with follow-up questions

        **Generate Quiz**: Create practice questions for a specific lecture — download as a .txt file

        **Summarize Lecture**: Get a structured summary of any lecture

        **Find in Textbook**: Locate topics across ISLR, ESL, or MML

        **Exam Prep**: Generate a structured study guide for any topic

        ## Indexed Materials
        - ST443 Lecture Slides (Lectures 1–10)
        - ISLR (Introduction to Statistical Learning)
        - Elements of Statistical Learning
        - Mathematics for Machine Learning

        ## Tips
        - Be specific in your questions for better results
        - Use the Chat tab to ask follow-up questions about an answer
        - Use quiz generation to test your understanding before exams
        """)

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
