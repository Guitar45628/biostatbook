import streamlit as st
from modules.nav import Navbar  # Assuming this is your navigation module
from modules.code_editor_all import code_editor_for_all
import numpy as np
from scipy.stats import binom
import io
from contextlib import redirect_stdout

# Page Titlebar
st.set_page_config(page_title='Week 02 | Probability and Probability Distributions') 

def section_1_content():
    st.header("Introduction to Probability")
    st.write("Probability is the branch of mathematics that deals with chance and uncertainty. It provides a way to measure how likely events are to occur. From everyday decisions to complex systems, probability helps us understand and manage risk and randomness.")
    st.subheader("Why Probability Matters?")
    st.write("- **Quantifying Uncertainty:** Probability allows us to express uncertainty numerically, making it possible to compare and rank different possibilities.")
    st.write("- **Making Informed Decisions:** By understanding the probabilities associated with different outcomes, we can make better decisions in situations involving uncertainty.")
    st.write("- **Analyzing Random Phenomena:** Probability provides tools for analyzing and modeling events that involve randomness, from coin flips to stock market fluctuations.")
    st.write("- **Application in Biostatistics:** Probability is fundamental to biostatistics, where it's used to analyze biological data, assess the effectiveness of treatments, and understand disease patterns.")

def section_2_content():
    st.header("Basic Probability Rules")
    st.subheader("Sample Space and Events")
    st.write("- **Sample Space (S):** The set of all possible outcomes of a random experiment.  *Example:* Rolling a single 6-sided die: S = {1, 2, 3, 4, 5, 6}")
    st.write("- **Event (E):** A specific subset of the sample space. *Example:* Event A = 'rolling an even number': E = {2, 4, 6}")
    st.subheader("Probability of an Event")
    st.write("The probability of an event E, denoted P(E), is a number between 0 and 1 that represents the likelihood of E occurring.")
    st.write("P(E) = (Number of favorable outcomes) / (Total number of possible outcomes)")
    st.subheader("Addition Rule")
    st.write("The Addition Rule is used to calculate the probability of either event A or event B occurring (or both).")
    st.write("- **Mutually Exclusive Events:** If A and B cannot occur at the same time, P(A∪B) = P(A) + P(B)")
    st.write("- **Non-Mutually Exclusive Events:** If A and B can occur at the same time, P(A∪B) = P(A) + P(B) - P(A∩B)")  # A∩B is the intersection of A and B
    # ... (Add more rules and explanations similarly, like Conditional Probability, Multiplication Rule, etc.)

def section_3_content():
    st.header("Probability Distributions")
    st.write("A Probability Distribution describes the likelihood of each possible outcome of a random variable. It can be discrete or continuous.")
    st.subheader("Discrete Probability Distributions")
    st.write("- **Binomial Distribution:**  Models the probability of a certain number of successes in a fixed number of independent trials.  *Example:*  Flipping a coin 10 times and counting the number of heads.")
    st.write("- **Poisson Distribution:** Models the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known average rate and independently of the time since the last event. *Example:* The number of customers arriving at a store in an hour.")
    st.subheader("Continuous Probability Distributions")
    st.write("- **Normal Distribution:** A bell-shaped distribution that is symmetrical around the mean. Many natural phenomena follow approximately a normal distribution. *Example:* Heights and weights of individuals.")
    # ... (Expand on each type with details from the PDF: formulas, properties, graphs)

def section_4_content():
    st.header("Applications in Biological Research")
    st.write("Probability is essential in biology due to the inherent variability in biological systems, the need to deal with uncertainty, and the ability to make predictions and inferences.")
    st.write("- **Genetics:**  Predicting the inheritance of traits, analyzing genetic variations.")
    st.write("- **Ecology:** Modeling population dynamics, studying species interactions.")
    st.write("- **Epidemiology:** Assessing disease risk, evaluating the effectiveness of public health interventions.")
    st.write("- **Bioinformatics:** Analyzing DNA and protein sequences, predicting gene function.")  # Added more detail
    # ... (Provide specific examples and connect to content from the PDF)

def section_table_of_contents():
    st.header("📚 Table of Contents")
    st.markdown("""
        <ol>
            <li><a href="#introduction-to-probability">Introduction to Probability</a></li>
            <li><a href="#basic-probability-rules">Basic Probability Rules</a></li>
            <li><a href="#probability-distributions">Probability Distributions</a></li>
            <li><a href="#applications-in-biological-research">Applications in Biological Research</a></li>
            <li><a href="#quiz-1">Quiz 1: Test Your Knowledge</a></li>
            <li><a href="#quiz-2">Quiz 2: Code Challenge</a></li>
            <li><a href="#quiz-3">Quiz 3: Probability Distributions</a></li>
        </ol>
    """, unsafe_allow_html=True)

def quiz_1():
    st.header("Quiz 1: Test Your Knowledge")

    questions = [
        {"question": "What is probability?",
         "options": ["-",
                     "A branch of mathematics that deals with chance and uncertainty", 
                     "A way to measure how likely events are to occur", 
                     "A tool for making decisions", 
                     "All of the above"],
         "answer": "All of the above"}, 

        {"question": "Which of the following is a reason why probability matters?",
         "options": ["-",
                     "Quantifying uncertainty", 
                     "Making informed decisions", 
                     "Analyzing random phenomena", 
                     "All of the above"],
         "answer": "All of the above"},

        {"question": "What is a sample space?",
         "options": ["-",
                     "The set of all possible outcomes of a random experiment", 
                     "Events that occur at the same time", 
                     "Events that cannot occur at the same time", 
                     "All of the above"],
         "answer": "The set of all possible outcomes of a random experiment"},

        {"question": "What is an event?",
         "options": ["-",
                     "A single possible result of an experiment", 
                     "A specific subset of the sample space", 
                     "Events that occur at the same time", 
                     "All of the above"],
         "answer": "A specific subset of the sample space"},

        {"question": "What does the addition rule calculate the probability of?",
         "options": ["-",
                     "Either event A or event B occurring (or both)", 
                     "Both event A and event B occurring", 
                     "Event A occurring given that event B has occurred", 
                     "None of the above"],
         "answer": "Either event A or event B occurring (or both)"},

        {"question": "What are independent events?",
         "options": ["-",
                     "Events where the outcome of one event does not affect the outcome of another event", 
                     "Events where the outcome of one event affects the outcome of another event", 
                     "Events that cannot occur at the same time", 
                     "None of the above"],
         "answer": "Events where the outcome of one event does not affect the outcome of another event"},

        {"question": "What is a probability distribution?",
         "options": ["-",
                     "A description of the likelihood of all possible outcomes of a random variable", 
                     "A function that assigns probabilities to each value of a random variable", 
                     "Both of the above"],
         "answer": "Both of the above"},

        {"question": "What type of random variables do discrete distributions apply to?",
         "options": ["-",
                     "Variables that can only take on specific values", 
                     "Variables that can take on any value within a range", 
                     "Both of the above"],
         "answer": "Variables that can only take on specific values"},

        {"question": "What type of random variables do continuous distributions apply to?",
         "options": ["-",
             "Variables that can only take on specific values ",
                     "Variables that can take on any value within a range", 
                     "Both of the above"],
         "answer": "Variables that can take on any value within a range"},

        {"question": "What is the Poisson distribution often used for?",
         "options": ["-",
                     "Modeling the number of events occurring in a given interval of time or space", 
                     "Modeling the probability of a certain number of successes in a fixed number of trials", 
                     "Modeling the distribution of data that is symmetrical around the mean"],
         "answer": "Modeling the number of events occurring in a given interval of time or space"}
    ]

    answers = {}
    score = 0
    submitted = False  # Flag to track if the quiz has been submitted

    for i, q in enumerate(questions):
        st.subheader(f"Question {i + 1} | {q['question']}")
        answer = st.selectbox("Select the correct answer", q["options"])
        answers[i] = answer

    if st.button("Submit Answers",use_container_width=True,type='secondary', key='submit-quiz-1'):
        submitted = True  # Set the flag to True when submitted
        for i, q in enumerate(questions):
            if answers[i] == q["answer"]:
                score += 1

        st.success(f"Your score: {score}/10")
        st.toast(f"Your score: {score}/10", icon='✅')
        st.balloons()

    if submitted:  # Only show answers and feedback after submission
        with st.expander("View Answers"):
            for i, q in enumerate(questions):
                st.write(f"**Question {i+1}:** {q['question']}")
                st.write(f"Your answer: {answers[i] or 'Not answered'}")  # Handle no answer
                correct = answers[i] == q['answer']
                st.write(f"Correct answer: {q['answer']}")
                if correct:
                    st.write(f"<span style='color:green;'>✅ Correct</span>", unsafe_allow_html=True)
                else:
                    st.write(f"<span style='color:red;'>❌ Incorrect</span>", unsafe_allow_html=True)
                if i < len(questions) - 1:
                    st.divider()


def quiz_2():
    st.header("Quiz 2: Code Challenge")

    # Mutually Exclusive Events
    st.subheader("Addition Rule Example | Mutually Exclusive Events")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("assets/week2-q2-1.png",width=100)
    st.write("We want to find the probability of rolling a dice and getting an even number or an odd number.")
    st.info("""
    **Event A:** Rolling an even number (E = {2, 4, 6}) 

    **Event B:** Rolling an odd number (E = {1, 3, 5})
    """)
    st.latex(r"P(A \cup B) = P(A) + P(B)")
    st.write("Let's calculate the probability using Python code:")
    code_mutually_exclusive = """
#Define the probability of rolling an even number
P_A = 3/6

#Define the probability of rolling an odd number
P_B = 3/6

#Calculate the probability of rolling an even or odd number
P_A_union_B = P_A + P_B

#Print the result
print('P(A ∪ B) =', P_A_union_B)
    """
    st.code(code_mutually_exclusive, language='python')
    if st.button("Run Code Mutually Exclusive", key='run-code-quiz-2-1', type='secondary',use_container_width=True):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code_mutually_exclusive)
        result = buf.getvalue()

        st.write("Output:")
        st.code(result, language='python')

    # Non-Mutually Exclusive Events
    st.subheader("Addition Rule Example | Non-Mutually Exclusive Events")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("assets/week2-q2-2.png",width=100)

    st.write("We want to find the probability of drawing a heart or a face card from a standard deck of 52 cards.")
    st.info("""
    **Event A:** Drawing a heart (13 cards) 

    **Event B:** Drawing a face card (Jack, Queen, King in each of the 4 suits = 12 cards) 

    **Event A and B:** Drawing a heart face card (3 cards: Jack, Queen, King of Hearts)
    """)

    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
    st.write("Let's calculate the probability using Python code:")
    code_non_mutually_exclusive = """
# Define the probabilities
P_A = 13/52  # Probability of drawing a heart
P_B = 12/52  # Probability of drawing a face card
P_A_intersect_B = 3/52  # Probability of drawing a heart face card

# Calculate the probability of drawing a heart or a face card
P_A_union_B = P_A + P_B - P_A_intersect_B

# Print the result
print('P(A ∪ B) =', P_A_union_B)
    """
    st.code(code_non_mutually_exclusive, language='python')
    if st.button("Run Code Non-Mutually Exclusive", key='run-code-quiz-2-2', type='secondary',use_container_width=True):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code_non_mutually_exclusive)
        result = buf.getvalue()

        st.write("Output:")
        st.code(result, language='python')

    # Addition Rules Challenge Question
    st.subheader("Addition Rules Challenge")

    st.write("You can use this codepad to calculate and check your answers for the following challenge questions.")
    
    # Mutually Exclusive Events Challenge Question
    st.subheader("Challenge 1")
    st.write("A student randomly chooses a day of the week to study. What is the probability that the student chooses a weekday (Monday-Friday) or the weekend (Saturday or Sunday)?")
    with st.expander("See Hint"):
        st.write("There are 5 weekdays and 2 weekend days in a week.")
        # add hint code with lib
        st.code("""
# Mutually Exclusive Events: P(A or B) = P(A) + P(B)
weekdays = 5
weekends = 2
total_days = weekdays + weekends

P_weekday = weekdays / total_days  # Probability of choosing a weekday
P_weekend = weekends / total_days  # Probability of choosing a weekend

P_weekday_or_weekend = P_weekday + P_weekend # Since they are mutually exclusive

print(f"Probability of choosing a weekday or weekend: {P_weekday_or_weekend}")
""", language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-challenge-1')
    user_answer_me = st.text_input("Enter your answer (as a fraction or decimal):", key="me_answer")
    correct_answer_me = 1  # Probability is 1 (or 7/7) since it covers all possibilities
    if st.button("Check Answer Mutually Exclusive", key='check-me', use_container_width=True):
        try:
            user_answer_me = eval(user_answer_me)  # Evaluate the expression
            if user_answer_me == correct_answer_me:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is {correct_answer_me}.")
        except (SyntaxError, NameError, TypeError):
            st.error("Invalid input. Please enter a valid fraction or decimal.")


    # Non-Mutually Exclusive Events Challenge Question
    st.subheader("Challenge 2")
    st.write("In a class of 30 students, 15 are in the math club and 10 are in the science club.  5 students are in both clubs. What is the probability that a randomly chosen student is in either the math club or the science club (or both)?")
    with st.expander("See Hint"):
        st.write("Use the formula for non-mutually exclusive events: P(A or B) = P(A) + P(B) - P(A and B)")
        st.code("""
import numpy as np
from scipy.stats import binom

# Non-Mutually Exclusive Events: P(A or B) = P(A) + P(B) - P(A and B)
total_students = 30
math_club = 15
science_club = 10
both_clubs = 5

P_math = math_club / total_students
P_science = science_club / total_students
P_both = both_clubs / total_students

P_math_or_science = P_math + P_science - P_both

print(f"Probability of being in math or science club (or both): {P_math_or_science}")
""", language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-challenge-2')
    user_answer_nme = st.text_input("Enter your answer (as a fraction or decimal):", key="nme_answer")
    correct_answer_nme = (15+10-5)/30  # Probability is (15+10-5)/30 = 20/30 = 2/3
    if st.button("Check Answer Non-Mutually Exclusive", key='check-nme', use_container_width=True):
        try:
            user_answer_value = eval(user_answer_nme)  # Evaluate the expression
            tol = 0.01  # Tolerance level (e.g., 0.01)
            if abs(user_answer_value - correct_answer_nme) < tol:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is {correct_answer_nme}.")
        except (SyntaxError, NameError, TypeError):
            st.error("Invalid input. Please enter a valid fraction or decimal.")
    
        # Multiplication Rule Examples
    st.subheader("Multiplication Rule Examples")

    st.write("Below are examples using the multiplication rule for calculating the probability of two events occurring together.")

    # Independent Events Example
    st.write("**Independent Events:**")
    st.write("Example: Rolling a 4 on a die and flipping a head on a coin.")
    st.latex(r"P(4 \text{ and } H) = P(4) \times P(H)")
    code_independent = """
# Independent Events:
P_die = 1/6   # Probability of rolling a 4 on a die
P_coin = 1/2  # Probability of flipping heads on a coin

P_4_and_H = P_die * P_coin

print('Probability of rolling a 4 and flipping heads:', P_4_and_H)
"""
    st.code(code_independent, language="python")
    if st.button("Run Independent Multiplication Rule", key='run-mult-indep', type='secondary', use_container_width=True):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code_independent)
        result = buf.getvalue()
        st.write("Output:")
        st.code(result, language="python")

    # Dependent Events Example
    st.write("**Dependent (Non-Independent) Events:**")
    st.write("Example: Drawing two aces in a row from a standard deck without replacement.")
    st.latex(r"P(\text{Ace}_1 \text{ and } \text{Ace}_2) = P(\text{Ace}_1) \times P(\text{Ace}_2|\text{Ace}_1)")
    code_dependent = """
# Dependent Events:
P_first_ace = 4/52  # Probability of drawing an ace first
P_second_ace_given_first = 3/51  # Probability of drawing a second ace given the first one was drawn

P_two_aces = P_first_ace * P_second_ace_given_first

print('Probability of drawing two aces in a row:', P_two_aces)
"""
    st.code(code_dependent, language="python")
    if st.button("Run Dependent Multiplication Rule", key='run-mult-dep', type='secondary', use_container_width=True):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code_dependent)
        result = buf.getvalue()
        st.write("Output:")
        st.code(result, language="python")
    
    # Challenge 3: Independent Multiplication Rule
    st.subheader("Challenge 3: Independent Multiplication Rule")
    st.write("A player rolls a fair die and flips a fair coin. What is the probability they roll a 6 and flip heads?")
    with st.expander("See Hint"):
        st.write("For independent events, the combined probability is the product of the individual probabilities.")
        st.code("""
# Probability of rolling a 6:
P_die = 1/6
# Probability of flipping heads:
P_coin = 1/2
# Combined probability:
P_result = P_die * P_coin  # 1/6 * 1/2 = 1/12
print('Probability:', P_result)
        """, language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-challenge-3')
    user_answer_ch3 = st.text_input("Enter your answer (as a fraction or decimal):", key="ch3_answer")
    correct_answer_ch3 = 1/12  # Approximately 0.08333...
    if st.button("Check Answer Challenge 3", key="check-ch3", use_container_width=True):
        try:
            user_answer_value = eval(user_answer_ch3)
            tol = 0.001  # Tolerance level
            if abs(user_answer_value - correct_answer_ch3) < tol:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is {correct_answer_ch3}.")
        except Exception:
            st.error("Invalid input. Please enter a valid fraction or decimal.")

    # Challenge 4: Dependent Multiplication Rule
    st.subheader("Challenge 4: Dependent Multiplication Rule")
    st.write("From a standard deck of 52 cards, what is the probability of drawing two hearts in a row without replacement?")
    with st.expander("See Hint"):
        st.write("For dependent events, the combined probability is the probability of the first event multiplied by the conditional probability of the second event given the first.")
        st.code("""
# Probability of drawing the first heart:
P_first_heart = 13/52
# Given the first is a heart, probability of drawing a second heart:
P_second_heart = 12/51
# Combined probability:
P_combined = P_first_heart * P_second_heart
print('Probability:', P_combined)
        """, language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-challenge-4')
    user_answer_ch4 = st.text_input("Enter your answer (as a fraction or decimal):", key="ch4_answer")
    correct_answer_ch4 = (13/52) * (12/51)
    if st.button("Check Answer Challenge 4", key="check-ch4", use_container_width=True):
        try:
            user_answer_value = eval(user_answer_ch4)
            tol = 0.001  # Tolerance level
            if abs(user_answer_value - correct_answer_ch4) < tol:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is {correct_answer_ch4}.")
        except Exception:
            st.error("Invalid input. Please enter a valid fraction or decimal.")

def quiz_3():
    st.header("Quiz 3: Probability Distributions")
    
    st.subheader("3.1 Discrete Probability Distributions")
    st.write("Answer each question below:")
    
    # Question 1: Binomial Distribution
    st.write("**Question 1 (Binomial Distribution):**")
    st.write("A fair coin is flipped 10 times. What is the probability of obtaining exactly 3 heads?")
    with st.expander("See Hint"):
        st.write("Use the binomial probability function. Hint: Use binom.pmf(3, 10, 0.5)")
        st.code("""
from scipy.stats import binom
P_exactly_3 = binom.pmf(3, 10, 0.5)
print('Probability:', P_exactly_3)
        """, language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-quiz3-binomial')
    user_answer_binom = st.text_input("Enter your answer for the Binomial question:", key="quiz3_binom")
    correct_answer_binom = 0.1171875
    if st.button("Check Answer (Binomial)", key="check-quiz3-binom", use_container_width=True):
        try:
            user_val = eval(user_answer_binom)
            tol = 0.001
            if abs(user_val - correct_answer_binom) < tol:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is approximately {correct_answer_binom}.")
        except Exception:
            st.error("Invalid input. Please enter a valid fraction or decimal.")
    
    st.divider()
    
    # Question 2: Poisson Distribution
    st.write("**Question 2 (Poisson Distribution):**")
    st.write("If the average number of events per period is 2, what is the probability that exactly 3 events occur in that period?")
    with st.expander("See Hint"):
        st.write("Use the Poisson probability formula: P(3) = (λ^3 * e^-λ) / 3! with λ=2.")
        st.code("""
import math
lambda_val = 2
P_exactly_3 = (lambda_val**3 * math.exp(-lambda_val)) / math.factorial(3)
print('Probability:', P_exactly_3)
        """, language="python")
    with st.expander("📝Use CodePad"):
        code_editor_for_all(key='codepad-quiz3-poisson')
    user_answer_poisson = st.text_input("Enter your answer for the Poisson question:", key="quiz3_poisson")
    correct_answer_poisson = 0.180447
    if st.button("Check Answer (Poisson)", key="check-quiz3-poisson", use_container_width=True):
        try:
            user_val = eval(user_answer_poisson)
            tol = 0.001
            if abs(user_val - correct_answer_poisson) < tol:
                st.success("Correct!")
                st.balloons()
            else:
                st.error(f"Incorrect. The correct answer is approximately {correct_answer_poisson}.")
        except Exception:
            st.error("Invalid input. Please enter a valid fraction or decimal.")
    
    st.subheader("3.2 Continuous Probability Distributions")
    st.write("**Question 3 (Normal Distribution Examples):**")
    st.write("Select which of the following examples are typically modeled by a normal distribution.")
   
    
    # Options using multi-select (checkbox-like)
    options = [
        "Human Height",                     # Correct
        "Blood Pressure",                   # Correct
        "IQ Score",                         # Correct
        "Monthly Rainfall",                 # Likely skewed
        "Number of Children in a Family",   # Discrete
        "Shoe Size",                        # Correct if considered normally distributed
        "Stock Market Returns",             # Often not normal
        "Daily Temperature",                # May be normal, but include as distractor
        "Annual Income"                     # Typically skewed
    ]
    # Define correct examples
    correct_options = {"Human Height", "Blood Pressure", "IQ Score", "Shoe Size"}
    
    user_selected = st.multiselect("Select all that apply:", options, key="quiz3_normal")
    
    if st.button("Check Answer (Normal Distribution)", key="check-quiz3-normal", use_container_width=True):
        if set(user_selected) == correct_options:
            st.success("Correct!")
            st.balloons()
        else:
            st.error("Incorrect. Please review which real-life measurements are typically modeled by a normal distribution.")

def main():
    # Add Navbar
    Navbar()

    # Title
    st.title('Week 02 | Probability and Probability Distributions')

    # Table of Contents
    section_table_of_contents()

    # Anchors should be defined BEFORE their respective sections are called
    st.markdown("<a id='introduction-to-probability'></a>", unsafe_allow_html=True) # Anchor for TOC
    section_1_content()

    st.markdown("<a id='basic-probability-rules'></a>", unsafe_allow_html=True) # Anchor for TOC
    section_2_content()

    st.markdown("<a id='probability-distributions'></a>", unsafe_allow_html=True) # Anchor for TOC
    section_3_content()
    
    st.markdown("<a id='applications-in-biological-research'></a>", unsafe_allow_html=True) # Anchor for TOC
    section_4_content()

    st.markdown("<a id='quiz-1'></a>", unsafe_allow_html=True)
    quiz_1()
    st.divider()

    st.markdown("<a id='quiz-2'></a>", unsafe_allow_html=True)
    quiz_2()
    st.divider()

    st.markdown("<a id='quiz-3'></a>", unsafe_allow_html=True)
    quiz_3()

    st.markdown("""
<div style='
    text-align: center;
    padding: 0px;
    margin: 0px auto;
    max-width: 600px;
    border-radius: 10px;
    ;
'>
    <h2>End of Week 2 🎉</h2>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()