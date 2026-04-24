import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import base64

# uv add langchain-ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# =========================================================
# CONFIG OLLAMA LLAVA
# =========================================================
llm = ChatOllama(
    model="llava",
    temperature=0.0
)

# =========================================================
# CHROMA DB + OPENCLIP
# =========================================================
chroma_client = chromadb.PersistentClient(path="vehicles-store-vdb")

image_loader = ImageLoader()
embedding_model = OpenCLIPEmbeddingFunction()

chroma_vdb = chroma_client.get_or_create_collection(
    name="vehicules",
    data_loader=image_loader,
    embedding_function=embedding_model,
)

# =========================================================
# LOAD DATA (IMAGES)
# =========================================================
def loadDataIntoVectorStore(folder_name):
    images_ids = []
    images_uris = []

    for index, file_name in enumerate(sorted(os.listdir(folder_name))):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_name, file_name)
            images_ids.append(str(index))
            images_uris.append(file_path)

    if len(images_ids) > 0:
        chroma_vdb.add(ids=images_ids, uris=images_uris)

# =========================================================
# SEARCH IMAGES
# =========================================================
def search_images(user_question):
    results = chroma_vdb.query(
        query_texts=[user_question],
        n_results=2,
        include=["uris", "distances"]
    )
    return results

# =========================================================
# ASK OLLAMA LLAVA (MULTIMODAL)
# =========================================================
def askLLM(user_question, images_result):

    if not images_result["uris"] or len(images_result["uris"][0]) < 2:
        return "Pas assez d'images trouvées."

    image_path1 = images_result["uris"][0][0]
    image_path2 = images_result["uris"][0][1]

    # affichage Streamlit
    st.image(image_path1, width=400)
    st.image(image_path2, width=400)

    # convertir en base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    img1_base64 = encode_image(image_path1)
    img2_base64 = encode_image(image_path2)

    # FORMAT LLAVA (IMPORTANT)
    message = HumanMessage(
        content=user_question,
        additional_kwargs={
            "images": [img1_base64, img2_base64]
        }
    )

    response = llm.invoke([message])
    return response.content

# =========================================================
# STREAMLIT APP
# =========================================================
def main():
    st.set_page_config(layout="wide")
    st.title("Multimodal RAG (Images + Ollama LLaVA)")

    # Charger les images UNE SEULE FOIS
    if "loaded" not in st.session_state:
        loadDataIntoVectorStore("images")
        st.session_state.loaded = True

    user_question = st.text_input("Pose ta question :")

    if user_question:
        results = search_images(user_question)
        response = askLLM(user_question, results)

        st.markdown("### Réponse")
        st.write(response)

# =========================================================
if __name__ == "__main__":
    main()