QUIZ_GENERATION = """Based ONLY on the content from {source}, generate {num_questions} exam-style theoretical questions.

Each question should:
- Test deep understanding of concepts, not just definitions.
- Reference specific formulas or methods from the lecture
- Be answerable using only the lecture content.
- If provided with context relating to previous exams, generate questions similar to those present in those papers. 

Example of a GOOD question:
"Explain why ridge regression shrinks coefficients but never sets them exactly to zero, while lasso can. What is the geometric interpretation?"

Example of a BAD question:
"What is regularization?"

Format your response as:
Q1: [question]
Q2: [question]
Q3: [question]
"""
