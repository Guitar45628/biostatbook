import datetime
from io import StringIO
import sys
import uuid  # added import for generating a unique key
from code_editor import code_editor
import streamlit as st


def code_editor_for_all(default_code="#Edit your code here!\n#Save before run!", key=None):
    # Generate a unique key if none is provided
    if key is None:
        key = str(uuid.uuid4())

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
        info=info_bar,
        key=key
    )

    # Default code if no user input
    user_code = response_dict.get("text", "# Default Python code goes here")

    st.warning(
        "*Don't forget to save your code before running it!* (ctrl+enter or save button)"
    )

    # Create unique keys for buttons based on the instance key
    run_key = f"{key}-run_codepad"
    download_key = f"{key}-download_codepad"

    result_output = None

    # สร้างคอลัมน์สำหรับปุ่ม Run และ Download
    col1, col2 = st.columns(2)
    with col1:
        # Run the code
        if st.button("Run Code", type="primary", use_container_width=True, key=run_key):
            try:
                # Capture the output
                output = StringIO()
                sys.stdout = output  # Redirect stdout
                exec(user_code)  # Execute the user code
                sys.stdout = sys.__stdout__  # Reset stdout
                result_output = output.getvalue()
                st.toast("Code executed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.download_button(
            label="Download my code",
            data=user_code,
            file_name=f"code-{int(datetime.datetime.now().timestamp())}.py",
            mime="text/x-python",
            type="tertiary",
            use_container_width=True,
            key=download_key
        )
    
    # แสดงผลลัพธ์เต็มความกว้างหลังคอลัมน์
    if result_output is not None:
        st.text_area("Output:", result_output, height=150)