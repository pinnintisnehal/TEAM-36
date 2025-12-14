import streamlit as st
import json
from agents import ask_drug, reminder_agent
from vectorstore import get_vectorstore, get_available_products
from guardrails import validate_product, validate_question
from agents import autonomous_ingestion

# Run autonomous pipeline
vectorstore = autonomous_ingestion()

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Medication Reminder Chatbot",
    page_icon="💊",
    layout="centered"
)

# -----------------------
# LOAD DB & PRODUCTS
# -----------------------
vectorstore = get_vectorstore()
products = get_available_products(vectorstore)

if not products:
    st.error("❌ No medications found in database. Please ingest data first.")
    st.stop()

# -----------------------
# SESSION STATE INIT
# -----------------------
if 'chats' not in st.session_state:
    st.session_state.chats = {}

if 'current_chat' not in st.session_state:
    st.session_state.current_chat = "Chat 1"
    st.session_state.chats["Chat 1"] = {
        "messages": [],
        "product": None
    }

# -----------------------
# SIDEBAR FOR CHATS
# -----------------------
st.sidebar.title("Chats")
if st.sidebar.button("New Chat"):
    new_id = len(st.session_state.chats) + 1
    name = f"Chat {new_id}"
    st.session_state.chats[name] = {
        "messages": [],
        "product": None
    }
    st.session_state.current_chat = name

for name in list(st.session_state.chats.keys()):
    if st.sidebar.button(name):
        st.session_state.current_chat = name

# -----------------------
# MAIN CHAT INTERFACE
# -----------------------
chat = st.session_state.chats[st.session_state.current_chat]
messages = chat["messages"]
product = chat["product"]

st.title("💊 Medication Reminder Chatbot")
st.caption("Label-aware assistant using Retrieval Augmented Generation (RAG)")

# Display chat messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Add welcome message if no messages
if product is None and len(messages) == 0:
    initial_msg = "Welcome to the Medication Reminder Chatbot! Please select a medication from the dropdown below to get started."
    with st.chat_message("assistant"):
        st.markdown(initial_msg)
    messages.append({"role": "assistant", "content": initial_msg})

# Conditional UI based on product selection
if product is None:
    product_options = ["-- Select a medication --"] + products
    selected = st.selectbox(
        "Choose a drug from the database",
        product_options,
        help="Medications are loaded dynamically from the vector database"
    )
    if selected != "-- Select a medication --":
        chat["product"] = selected
        select_msg = f"Selected Medication: **{selected}**. You can ask questions about its label, or type 'reminder' to generate a reminder schedule. Commands: /change to change medication, /quit to reset chat."
        with st.chat_message("assistant"):
            st.markdown(select_msg)
        messages.append({"role": "assistant", "content": select_msg})
        st.rerun()
else:
    # Action buttons for quick options
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate Reminder"):
            try:
                validate_product(product, products)
                with st.spinner("Creating reminder schedule..."):
                    reminder = reminder_agent(product)
                reminder_content = "Reminder Schedule:\n```json\n" + json.dumps(reminder, indent=2) + "\n```"
                with st.chat_message("assistant"):
                    st.markdown(reminder_content)
                messages.append({"role": "assistant", "content": reminder_content})
            except Exception as e:
                error_msg = str(e)
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

    with col2:
        if st.button("Change Drug"):
            chat["product"] = None
            change_msg = "Okay, let's change the medication."
            with st.chat_message("assistant"):
                st.markdown(change_msg)
            messages.append({"role": "assistant", "content": change_msg})
            st.rerun()

    with col3:
        if st.button("Quit Chat"):
            chat["messages"] = []
            chat["product"] = None
            st.success("Chat reset!")
            st.rerun()

    prompt = st.chat_input("Type your message here (or use buttons above)...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        try:
            validate_product(product, products)

            prompt_lower = prompt.lower().strip()
            if prompt_lower == "/change":
                chat["product"] = None
                response = "Okay, changing medication."
                with st.chat_message("assistant"):
                    st.markdown(response)
                messages.append({"role": "assistant", "content": response})
                st.rerun()
            elif prompt_lower == "/quit":
                chat["messages"] = []
                chat["product"] = None
                response = "Chat reset!"
                with st.chat_message("assistant"):
                    st.markdown(response)
                messages.append({"role": "assistant", "content": response})
                st.rerun()
            elif prompt_lower == "reminder":
                with st.spinner("Creating reminder schedule..."):
                    reminder = reminder_agent(product)
                reminder_content = "Reminder Schedule:\n```json\n" + json.dumps(reminder, indent=2) + "\n```"
                with st.chat_message("assistant"):
                    st.markdown(reminder_content)
                messages.append({"role": "assistant", "content": reminder_content})
            else:
                validate_question(prompt)
                with st.spinner("Searching drug label..."):
                    answer = ask_drug(product, prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = str(e)
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            messages.append({"role": "assistant", "content": error_msg})

# -----------------------
# FOOTER
# -----------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with Streamlit • ChromaDB • Ollama • RAG
    </p>
    """,
    unsafe_allow_html=True
)
