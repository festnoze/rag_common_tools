
import uuid
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

class RagChunking:
    @staticmethod
    def split_text_into_chunks(documents: list, chunk_size: int = 2000, chunk_overlap: int = 100, max_chunk_size: int = 5461) -> list[Document]:
        all_chunks = []
        txt_splitter = RagChunking._get_text_splitter(chunk_size, chunk_overlap)
        for document in documents:
            if not document:
                continue
            if isinstance(document, dict):
                document = Document(page_content=document.get('page_content', ''), metadata=document.get('metadata', {}))
            chunks_content = txt_splitter.split_text(document.page_content)
            doc_chunks = [Document(page_content=chunk, metadata=document.metadata) for chunk in chunks_content]
            
            # Set each chunk metadata 'id' to a new UUID, or to 'doc_id' if single chunk
            if len(doc_chunks) == 1:
                doc_chunks[0].metadata['id'] = document.metadata.get('doc_id', str(uuid.uuid4()))
            else:
                for chunk in doc_chunks:
                    chunk.metadata['id'] = str(uuid.uuid4())
            all_chunks.extend(doc_chunks)

        # Ensure chunks do not exceed the maximum allowed size
        valid_chunks = []
        for chunk in all_chunks:
            if len(chunk.page_content.split(' ')) <= max_chunk_size*1.15:
                valid_chunks.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
            else:
                # Optionally, you can further split the chunk here if it exceeds the max size
                smaller_chunks = RagChunking.split_text_with_overlap(chunk.page_content, chunk_size, chunk_overlap)
                for small_chunk in smaller_chunks:
                    if len(small_chunk) <= max_chunk_size:
                        valid_chunks.append(Document(page_content=small_chunk, metadata=chunk.metadata))
        return valid_chunks

    @staticmethod
    def _get_text_splitter(chunk_size, chunk_overlap):
        txt_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\r\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )        
        return txt_splitter
    
    @staticmethod
    def split_text_with_overlap(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """ Splits a string into chunks of specified size with overlap. """
        start = 0
        chunks = []
        if chunk_size <= 0: raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap >= chunk_size: raise ValueError("chunk_overlap must be smaller than chunk_size.")

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks
       