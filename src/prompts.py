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

SUMMARIZE_LECTURE ="""Summarize the retrieved lecture content by organizing into these sections:

1. **Key Concepts**: 
   - List all main ideas and definitions from the lecture
   - Explain each concept in 2-3 sentences
   - Include the context and why each concept matters

2. **Mathematical Formulas**: 
   - Write out ALL important equations mentioned in the lecture.
   - Explain what each variable/symbol means
   - Describe when and why to use each formula
   - Use the exact notation from the lecture slides

3. **Examples/Applications**:
   - Describe any examples, case studies, or applications mentioned
   - Explain what each example illustrates
   - Include any datasets or scenarios used

4. **Important Points for Exams**:
   - Key formulas to memorize
   - Critical conceptual understanding points
   - Common pitfalls or mistakes to avoid
   - Connections to other topics

Be thorough and detailed. This is for exam preparation, so include everything important from the lecture.
If formulas are present, write them out completely. If examples are given, describe them fully.
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