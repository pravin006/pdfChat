import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

def getPdfText(pdfDocs):
    text =''
    for pdf in pdfDocs:
        pdfReader = PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text        

def getTextChucks(rawText):
    textSplitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = textSplitter.split_text(rawText)
    return chunks

def getVectorStore(textChunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl' )
    vectorStore = FAISS.from_texts(texts=textChunks, embedding=embeddings)
    return vectorStore

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Chat with multiple PDFs")
    st.text_input("Ask a question")

    with st.sidebar:
        st.subheader("Your documents")
        pdfDocs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner('Processing'):
                rawText = getPdfText(pdfDocs)
                # st.write(rawText)

                textChunks = getTextChucks(rawText)
                # st.write(textChunks)

                vectorStore = getVectorStore(textChunks)


if __name__ == '__main__':
    main()