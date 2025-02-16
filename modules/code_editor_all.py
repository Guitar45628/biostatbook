from io import StringIO
import sys
from code_editor import code_editor
import streamlit as st


def code_editor_for_all(default_code="#Edit your code here!\n#Save before run!"):
    st.subheader("Edit and Run the Code here:")
    # CSS to inject related to info bar
    css_string = '''
    background-color: #bee1e5;

    body > #root .ace-streamlit-dark~& {
        background-color: #262830;
    }

    .ace-streamlit-dark~& span {
        color: #fff;
        opacity: 0.6;
    }

    span {
        color: #000;
        opacity: 0.5;
    }

    .code_editor-info.message {
        width: inherit;
        margin-right: 75px;
        order: 2;
        text-align: center;
        opacity: 0;
        transition: opacity 0.7s ease-out;
    }

    .code_editor-info.message.show {
        opacity: 0.6;
    }

    .ace-streamlit-dark~& .code_editor-info.message.show {
        opacity: 0.5;
    }
    '''

    # Create info bar dictionary
    info_bar = {
        "name": "language info",
        "css": css_string,
        "style": {
            "order": "1",
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "width": "100%",
            "height": "2.5rem",
            "padding": "0rem 0.75rem",
            "borderRadius": "8px 8px 0px 0px",
            "zIndex": "9993"
        },
        "info": [{
            "name": "Python",
            "style": {"width": "100px"}
        }]
    }

    # Add info bar to code editor
    response_dict = code_editor(
        default_code, 
        height=[10, 20], 
        buttons=[{
            "name": "Save",
            "feather": "Save",
            "hasText": True,
            "commands": ["save-state", ["response", "saved"]],
            "response": "saved",
            "showWithIcon": True,
            "style": {"top": "0rem", "right": "0.4rem"}
        }],
        lang="python", 
        info=info_bar
    )

    # Default code if no user input
    user_code = response_dict.get("text", "# Default Python code goes here")

    st.write(
        "*Don't forget to save your code before running it!* (ctrl+enter or save button)"
    )

    # Run the code
    if st.button("Run Code"):
        try:
            # Capture the output
            output = StringIO()
            sys.stdout = output  # Redirect stdout
            exec(user_code)  # Execute the user code
            sys.stdout = sys.__stdout__  # Reset stdout
            st.toast("Code executed successfully!")
            st.text_area("Output:", output.getvalue(), height=150)

        except Exception as e:
            st.error(f"Error: {e}")

