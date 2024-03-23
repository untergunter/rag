from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from typing import List
import chromadb

class DB:
    def __init__(self,
                 path:str='dbpath',
                 collection_name:str='toy_rag'):

        self.path = path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.id = 0

        self.splitter = CharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=100)

    def __pop_next_id__(self):
        """
        This function returns the next id and increments it by 1
        :return:
        """
        val = str(self.id)
        self.id += 1
        return val

    def __add_to_db__(self,
                      document:str,
                      metadata:dict):
        """
        This function adds a single pair of document and metadata to the database
        :param document:
        :param metadata:
        :return:
        """
        self.collection.add(metadatas=[metadata],
                            documents=[document],
                            ids=[self.__pop_next_id__()])

    def add_pdf(self,
                path: str) -> None:
        """
        converts the pdf to string, parses it and inserts to the database
        :param path:
        :return:
        """
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        docs = self.splitter.split_documents(pages)
        for doc in docs:
            metadata = doc.metadata
            text_content = doc.page_content
            self.__add_to_db__(
                document=text_content,
                metadata = metadata)

    def search(self,query:str,n_results:int) -> list:
        results = self.collection.query(query_texts=[query],
                                        n_results=n_results)
        return results

    def add_pdf_folder(self,
                       path: str) -> None:
        #TODO
        pass

class Answer:
    def __init__(self):
        self.model = Ollama(model='phi')

    def use_text_to_answer(self,
                           query,
                           texts:List[str]):
        texts_seperated_by_line = "\n".join(texts)
        template = f'{query} Base your answer on the following texts only:{texts_seperated_by_line}'
        answer = self.model.invoke(template)
        return answer

if __name__ == '__main__':
    db = DB()
    asistent = Answer()

    pdf_path = '/home/ido/data/idc/forest_sensing/papers/nyborg 2022.pdf'
    db.add_pdf(pdf_path)

    question = 'what remote sensing instrument is used and by which writer?'
    results = db.search(query=question,
                    n_results=5)
    texts = results['documents'][0]
    for text in texts:
        print(text)
    answer = asistent.use_text_to_answer(question,texts)
    print(answer)

