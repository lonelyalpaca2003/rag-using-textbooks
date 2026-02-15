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

SUMMARIZE_LECTURE = """Summarize the lecture content given in {lecture} notes by organizing into these sections:

1. **Key Concepts**: Main ideas and definitions
2. **Mathematical Formulas**: Important equations with explanations
3. **Examples/Applications**: Specific examples used
4. **Important Points for Exams**: Critical things to remember

Be concise but complete. Use the specific notation from the lecture.
"""

FIND_TEXTBOOK_PAGES = """Find the page numbers from the {textbook} for content that corresponds most with {topic}. For each topic
provide 
- specific page number or range of pages
- Brief description of what's on those pages

If you are unsure or can't find the topic in the book, express that it is not possible.
Format:
**{topic}**: Pages X-Y - [description]
"""

EXAM_PREP = """Based on the retrieved lecture content, create an exam preparation guide for {topic}.

Include:
1. **Core concepts to memorize**
2. **Key formulas** (with variable definitions)
3. **Common mistakes to avoid**
4. **Likely exam question types**
5. **Quick review checklist**
"""