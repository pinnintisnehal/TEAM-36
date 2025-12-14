import streamlit as st
import json
from agents import ask_drug, reminder_agent
from vectorstore import get_vectorstore, get_available_products
from guardrails import validate_product, validate_question
from agents import autonomous_ingestion
from ocr_utils import identify_drug_name

# Run autonomous pipeline (only once ideally, but okay here for demo)
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
    st.rerun()

for name in list(st.session_state.chats.keys()):
    if st.sidebar.button(name):
        st.session_state.current_chat = name
        st.rerun()

# -----------------------
# MAIN CHAT INTERFACE
# -----------------------
chat = st.session_state.chats[st.session_state.current_chat]
messages = chat["messages"]
product = chat["product"]

st.title("💊 Medication Reminder Chatbot")
st.caption("Label-aware assistant using Retrieval Augmented Generation (RAG)")

# Display chat history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome message if empty
if len(messages) == 0 and product is None:
    welcome_msg = "Welcome to the Medication Reminder Chatbot! Upload an image of your medication or select one from the dropdown below."
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)
    messages.append({"role": "assistant", "content": welcome_msg})

# -----------------------
# IMAGE UPLOAD & OCR SECTION (Always shown if no product selected)
# -----------------------
if product is None:
    st.subheader("📷 Upload Medication Image (Optional)")
    uploaded_image = st.file_uploader(
        "Upload a clear photo of your drug packaging/label",
        type=["jpg", "png", "jpeg"],
        key="uploader"  # Avoid duplicate widget issues
    )

    drug_identified = False

    if uploaded_image:
        with st.spinner("Analyzing image..."):
            drug_name, ocr_text = identify_drug_name(uploaded_image, products)

        st.subheader("Extracted Text")
        st.code(ocr_text)

        if drug_name:
            st.success(f"✅ Detected Drug: **{drug_name}**")
            chat["product"] = drug_name

            select_msg = f"Great! I've selected **{drug_name}** based on your image.\n\nYou can now ask questions about its label, or type **'reminder'** (or click the button) to generate a schedule.\n\nCommands: `/change` to switch drug, `/quit` to reset."
            with st.chat_message("assistant"):
                st.markdown(select_msg)
            messages.append({"role": "assistant", "content": select_msg})
            st.rerun()
        else:
            st.warning("⚠️ Could not confidently identify the drug from the image.")
            st.info("👇 Please select your medication manually from the dropdown below.")

    # -----------------------
    # DROPDOWN FALLBACK (Only if no product yet)
    # -----------------------
    if product is None:  # Still None after possible OCR attempt
        st.subheader("📋 Or Select Medication Manually")
        product_options = ["-- Select a medication --"] + products
        selected = st.selectbox(
            "Choose your drug",
            product_options,
            index=0,
            help="Medications loaded from the database"
        )

        if selected != "-- Select a medication --":
            chat["product"] = selected
            select_msg = f"Selected Medication: **{selected}**.\n\nYou can now ask questions about its label, or type **'reminder'** to generate a schedule.\n\nCommands: `/change` to switch, `/quit` to reset."
            with st.chat_message("assistant"):
                st.markdown(select_msg)
            messages.append({"role": "assistant", "content": select_msg})
            st.rerun()

# -----------------------
# MAIN CHAT (When product IS selected)
# -----------------------
else:
    st.info(f"Current Medication: **{product}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Reminder"):
            with st.spinner("Generating reminder schedule..."):
                try:
                    validate_product(product, products)
                    reminder = reminder_agent(product)
                    reminder_content = "🔔 **Reminder Schedule**\n```json\n" + json.dumps(reminder, indent=2) + "\n```"
                    with st.chat_message("assistant"):
                        st.markdown(reminder_content)
                    messages.append({"role": "assistant", "content": reminder_content})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    with st.chat_message("assistant"):
                        st.markdown(error_msg)
                    messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

    with col2:
        if st.button("Change Drug"):
            chat["product"] = None
            change_msg = "Medication cleared. Please upload an image or select a new one."
            with st.chat_message("assistant"):
                st.markdown(change_msg)
            messages.append({"role": "assistant", "content": change_msg})
            st.rerun()

    with col3:
        if st.button("Quit Chat"):
            chat["messages"] = []
            chat["product"] = None
            st.success("Chat reset successfully!")
            st.rerun()

    # Chat input
    prompt = st.chat_input("Ask about dosage, side effects, or type 'reminder'...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        try:
            validate_product(product, products)
            prompt_lower = prompt.lower().strip()

            if prompt_lower == "/change":
                chat["product"] = None
                response = "Medication cleared. Please choose a new one."
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
                with st.spinner("Generating reminder..."):
                    reminder = reminder_agent(product)
                    reminder_content = "🔔 **Reminder Schedule**\n```json\n" + json.dumps(reminder, indent=2) + "\n```"
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
            error_msg = f"Error: {str(e)}"
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
    Built with Streamlit • ChromaDB • Ollama • RAG • OCR
    </p>
    """,
    unsafe_allow_html=True
)
