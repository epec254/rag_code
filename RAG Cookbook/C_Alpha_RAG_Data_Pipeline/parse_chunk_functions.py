# Databricks notebook source
# MAGIC %md ## Configuration & setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

from typing import List, Dict, Tuple
import warnings
from abc import ABC, abstractmethod
from typing import List, TypedDict

DEBUG = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embedding model constants

# COMMAND ----------

class EmbeddingModelConfig(TypedDict):
    endpoint: str
    model_name: str
    
EMBEDDING_MODELS = {
    "Alibaba-NLP/gte-large-en-v1.5": {
        "context_window": 8192,
        "tokenizer": "hugging_face",
        "type": "custom",
    },
    "nomic-ai/nomic-embed-text-v1": {
        "context_window": 8192,
        "tokenizer": "hugging_face",
        "type": "custom",
    },
    "BAAI/bge-large-en-v1.5": {
        "context_window": 512,
        "tokenizer": "hugging_face",
        "type": "FMAPI",
    },
    "text-embedding-ada-002": {"context_window": 8192, "tokenizer": "tiktoken"},
    "text-embedding-3-small": {"context_window": 8192, "tokenizer": "tiktoken"},
    "text-embedding-3-large": {"context_window": 8192, "tokenizer": "tiktoken"},
}

# COMMAND ----------

# MAGIC %md ### Column name constants

# COMMAND ----------

# # Bronze table
# DOC_URI_COL_NAME = "doc_uri"
# CONTENT_COL_NAME = "raw_doc_contents_string"
# BYTES_COL_NAME = "raw_doc_contents_bytes"
# BYTES_LENGTH_COL_NAME = "raw_doc_bytes_length"
# MODIFICATION_TIME_COL_NAME = "raw_doc_modification_time"

# # Bronze table auto loader names
# LOADER_DEFAULT_DOC_URI_COL_NAME = "path"
# LOADER_DEFAULT_BYTES_COL_NAME = "content"
# LOADER_DEFAULT_BYTES_LENGTH_COL_NAME = "length"
# LOADER_DEFAULT_MODIFICATION_TIME_COL_NAME = "modificationTime"

# # Silver table
# PARSED_OUTPUT_STRUCT_COL_NAME = "parser_output"
# PARSED_OUTPUT_CONTENT_COL_NAME = "doc_parsed_contents"
# PARSED_OUTPUT_STATUS_COL_NAME = "parser_status"
# PARSED_OUTPUT_METADATA_COL_NAME = "parser_metadata"

# # Gold table

# # intermediate values
# CHUNKED_OUTPUT_STRUCT_COL_NAME = "chunker_output"
# CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME = "chunked_texts"
# CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME = "chunker_status"
# CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME = "chunker_metadata"

# FULL_DOC_PARSED_OUTPUT_COL_NAME = "parent_doc_parsed_contents"
# CHUNK_TEXT_COL_NAME = "chunk_text"
# CHUNK_ID_COL_NAME = "chunk_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parsing Functions
# MAGIC
# MAGIC Each parsing function is defined as a implementation of the `FileParser` abstract class.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Abstract `FileParser` class

# COMMAND ----------

from abc import ABC, abstractmethod


class ParserReturnValue(TypedDict):
    PARSED_OUTPUT_CONTENT_COL_NAME: str
    PARSED_OUTPUT_STATUS_COL_NAME: str


class FileParser(ABC):
    """
    Abstract base class for file parsing. Implementations of this class are designed to parse documents and return the parsed content as a string.
    """

    def __init__(self):
        """
        Initializes the FileParser instance.
        If your strategy can be tuned with parameters, implement this function e.g., __init__(param1="default_value"), etc
        """
        pass

    def __str__(self):
        """
        Provides a generic string representation of the instance, including the class name and its relevant parameters.
        Do not implement unless you want to control how the strategy is dumped to the RAG configuration YAML.
        """
        # Assuming all relevant parameters are stored as instance attributes
        params_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{type(self).__name__}({params_str})"

    @abstractmethod
    def supported_file_extensions(self) -> List[str]:
        """
        List of file extensions supported by this parser.

        Returns:
            List[str]: A list of supported file extensions.
        """
        return []

    def required_pip_packages(self) -> List[str]:
        """
        Array of packages to install via `%pip install package1 package2`
        """
        return []

    def required_aptget_packages(self) -> List[str]:
        """
        Array of packages to install via `sudo apt-get install package1 package2`
        """
        return []

    def load(self) -> bool:
        """
        Called before the parser is used to load any necessary configuration/models/etc.
        Returns True on success, False otherwise.
        You can assume all packages defined in `required_pip_packages` and `required_aptget_packages` are installed.
        For example, you might load a model from HuggingFace here.
        """
        return True

    @abstractmethod
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> ParserReturnValue:
        """
        Parses the document content (passed as bytes) and returns the parsed content.

        Parameters:
            raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed.

        Returns:
            ParserReturnValue: A dictionary containing the parsed content and status.
        """
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: "parsed_contents_as_string",
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }

    #@abstractmethod
    # TODO: Remove the need for this by adjusting the delta table pipeline to convert the strings into bytes
    def parse_string(
        self,
        raw_doc_contents_string: str,
    ) -> ParserReturnValue:
        """
        Parses the document content (passed as a string) and returns the parsed content.  

        Parameters:
            raw_doc_contents_string (str): The string of the document to be parsed.

        Returns:
            ParserReturnValue: A dictionary containing the parsed content and status.
        """
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: "parsed_contents_as_string",
            PARSED_OUTPUT_STATUS_COL_NAME: "ERROR: parse_string not implemented",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### HTML & Markdown

# COMMAND ----------

# MAGIC %md
# MAGIC #### HTMLToMarkdownify
# MAGIC
# MAGIC Convert HTML to Markdown via `markdownify` library.

# COMMAND ----------

class HTMLToMarkdownify(FileParser):
    def load(self) -> bool:
        return True

    def required_pip_packages(self) -> List[str]:
        return ["markdownify"]
    
    def required_aptget_packages(self) -> List[str]:
        return []
    
    def supported_file_extensions(self):
        return ["html"]

    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        from markdownify import markdownify as md

        markdown = md(raw_doc_contents_bytes.decode("utf-8"))
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: markdown.strip(),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }

    def parse_string(
        self,
        raw_doc_contents_string: str,
    ) -> Dict[str, str]:
        from markdownify import markdownify as md

        markdown = md(raw_doc_contents_string)
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: markdown.strip(),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
if DEBUG:
    parser = HTMLToMarkdownify()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md #### PassThroughNoParsing
# MAGIC
# MAGIC Decode the bytes and return the resulting string, stripped of trailing/leading whitespace.  Intended for use with `txt`, `markdown` or `html` files where parsing is not required.

# COMMAND ----------

class PassThroughNoParsing(FileParser):
    def load(self) -> bool:
        return True
    
    def required_pip_packages(self) -> List[str]:
        return []
    
    def required_aptget_packages(self) -> List[str]:
        return []

    def supported_file_extensions(self):
        return ["html", "txt", "md"]
    
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        text = raw_doc_contents_bytes.decode("utf-8")

        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: text.strip(),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }
    
    def parse_string(
        self,
        raw_doc_contents_string: bytes,
    ) -> Dict[str, str]:
        text = raw_doc_contents_string

        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: text.strip(),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
if DEBUG:
    parser = PassThroughNoParsing()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### PDF

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyMuPdfMarkdown
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library, converting the output to Markdown.

# COMMAND ----------

class PyMuPdfMarkdown(FileParser):
    def load(self) -> bool:
        return True

    def required_pip_packages(self) -> List[str]:
        return ["pymupdf", "pymupdf4llm"]
    
    def required_aptget_packages(self) -> List[str]:
        return []
    
    def supported_file_extensions(self):
        return ["pdf"]
    
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        import fitz
        import pymupdf4llm

        pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(pdf_doc)

        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: md_text.strip(),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
if DEBUG:
    parser = PyMuPdfMarkdown()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyMuPdf
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library.

# COMMAND ----------

class PyMuPdf(FileParser):
    def load(self) -> bool:
        return True

    def required_pip_packages(self) -> List[str]:
        return ["pymupdf"]
    
    def required_aptget_packages(self) -> List[str]:
        return []
    
    def supported_file_extensions(self):
        return ["pdf"]
    
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        import fitz

        pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
        output_text = [page.get_text() for page in pdf_doc]

        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: "\n".join(output_text),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
if DEBUG:
    parser = PyMuPdf()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### PyPdf
# MAGIC
# MAGIC Parse a PDF with `pypdf` library.

# COMMAND ----------

class PyPdf(FileParser):
    def load(self) -> bool:
        return True

    def required_pip_packages(self) -> List[str]:
        return ["pypdf"]
    
    def required_aptget_packages(self) -> List[str]:
        return []
    
    def supported_file_extensions(self):
        return ["pdf"]
    
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        from pypdf import PdfReader
        import io

        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)

        output_text = [page_content.extract_text() for page_content in reader.pages]

        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: "\n".join(output_text),
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
if DEBUG:
    parser = PyPdf()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### UnstructuredPDF
# MAGIC
# MAGIC Parse a PDF file with `unstructured` library. Defaults to using the `hi_res` strategy with the `yolox` model.
# MAGIC
# MAGIC TODO: This parser runs for 10 mins and still doesn't complete.  Debug & fix

# COMMAND ----------

class UnstructuredPDF(FileParser):
    def __init__(self, strategy="hi_res", hi_res_model_name="yolox"):
        """
        Initializes an instance of the UnstructuredPDF class with a specified document parsing strategy and high-resolution model name.

        Parameters:
        - strategy (str): The strategy to use for parsing the PDF document. Options include:
            - "ocr_only": Runs the document through Tesseract for OCR and then processes the raw text. Recommended for documents with multiple columns that do not have extractable text. Falls back to "fast" if Tesseract is not available and the document has extractable text.
            - "fast": Extracts text using pdfminer and processes the raw text. Recommended for most cases where the PDF has extractable text. Falls back to "ocr_only" if the text is not extractable.
            - "hi_res": Identifies the layout of the document using a specified model (e.g., detectron2_onnx). Uses the document layout to gain additional information about document elements. Recommended if your use case is highly sensitive to correct classifications for document elements. Falls back to "ocr_only" if the specified model is not available.
          The default strategy is "hi_res".
        - hi_res_model_name (str): The name of the model to use for the "hi_res" strategy. Options include:
            - "detectron2_onnx": A Computer Vision model by Facebook AI that provides object detection and segmentation algorithms with ONNX Runtime. It is the fastest model for the "hi_res" strategy.
            - "yolox": A single-stage real-time object detector that modifies YOLOv3 with a DarkNet53 backbone.
            - "yolox_quantized": Runs faster than YoloX and its speed is closer to Detectron2.
          The default model is "yolox".
        """
        if strategy not in ('ocr_only', 'hi_res', 'fast'):
            raise ValueError(f"strategy must be one of 'ocr_only', 'hi_res', 'fast'")
        if strategy == 'hi_res' and hi_res_model_name not in ('yolox', 'yolox_quantized','detectron2_onnx'):
            raise ValueError(f"hi_res_model_name must be one of 'yolox', 'yolox_quantized', 'detectron2_onnx'")
        self.strategy = strategy
        self.hi_res_model_name = hi_res_model_name
        
    def required_pip_packages(self) -> List[str]:
        return ["markdownify", '"unstructured[local-inference, all-docs]"', "pdfminer", "nltk"]
    
    def required_aptget_packages(self) -> List[str]:
        return ["poppler-utils", "tesseract-ocr"]
    
    def supported_file_extensions(self):
        return ["pdf"]
    
    def load(self) -> bool:
        try:
            import nltk
            from unstructured_inference.models.base import get_model

            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")

            model = get_model(self.hi_res_model_name)
            return True
        except Exception as e:
            return False

    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        from unstructured.partition.pdf import partition_pdf
        import io
        from markdownify import markdownify as md

        sections = partition_pdf(
            file=io.BytesIO(raw_doc_contents_bytes),
            strategy=self.strategy,  # mandatory to use ``hi_res`` strategy
            extract_images_in_pdf=True,  # mandatory to set as ``True``
            extract_image_block_types=["Image", "Table"],  # optional
            extract_image_block_to_payload=False,  # optional
            hi_res_model_name=self.hi_res_model_name, 
            infer_table_structure=True,
        )
        text_content = ""
        for section in sections:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += section.text
                else:
                    text_content += section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += section.text
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: text_content,
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }


# Test the function on 1 row
# TODO: Make it work with hi_res - right now, it is very slow.
if DEBUG:
    parser = UnstructuredPDF(strategy="fast")
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### DocX

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyPandocDocx
# MAGIC
# MAGIC Parse a DocX file with Pandoc parser using the `pypandoc` library

# COMMAND ----------

class PyPandocDocx(FileParser):
    def load(self) -> bool:
        return True

    def supported_file_extensions(self):
        return ["docx"]
    
    def required_pip_packages(self) -> List[str]:
        return ["pypandoc_binary"]
    
    def required_aptget_packages(self) -> List[str]:
        return ["pandoc"]
    
    def parse_bytes(
        self,
        raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        import pypandoc
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(raw_doc_contents_bytes)
            temp_file_path = temp_file.name
            md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

            return {
                PARSED_OUTPUT_CONTENT_COL_NAME: md,
                PARSED_OUTPUT_STATUS_COL_NAME: f"SUCCESS",
            }


# Test the function on 1 row
# TODO: Make it work with hi_res - right now, it is very slow.
if DEBUG:
    parser = PyPandocDocx()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md #### UnstructuredDocX
# MAGIC
# MAGIC Parse a DocX file with the `unstructured` library.

# COMMAND ----------

class UnstructuredDocX(FileParser):
    def load(self) -> bool:
        try:
            import nltk
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            return True
        except Exception as e:
            print(e)
            return False
    
    def required_pip_packages(self) -> List[str]:
        return ["markdownify", '"unstructured[local-inference, all-docs]"', "pdfminer", "nltk"]
    
    def required_aptget_packages(self) -> List[str]:
        return ["pandoc"]
    
    def supported_file_extensions(self):
        return ["docx"]
    
    def parse_bytes(
        self, raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        from unstructured.partition.docx import convert_and_partition_docx
        import io
        from markdownify import markdownify as md

        sections = convert_and_partition_docx(
            file=io.BytesIO(raw_doc_contents_bytes),
            source_format="docx"
        )
        text_content = ""
        for section in sections:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += section.text
                else:
                    text_content += section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += section.text
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: text_content,
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }

# Test the function on 1 row
if DEBUG:
    parser = UnstructuredDocX()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### PPTX

# COMMAND ----------

# MAGIC %md #### UnstructuredPPTX
# MAGIC
# MAGIC Parse a PPTX file with the `unstructured` library.

# COMMAND ----------


class UnstructuredPPTX(FileParser):
    def load(self) -> bool:
        try:
            import nltk
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            return True
        except Exception as e:
            print(e)
            return False

    def supported_file_extensions(self):
        return ["pptx"]
    
    def required_pip_packages(self) -> List[str]:
        return ["markdownify", '"unstructured[local-inference, all-docs]"']
    
    def required_aptget_packages(self) -> List[str]:
        return []
    
    def parse_bytes(
        self, raw_doc_contents_bytes: bytes,
    ) -> Dict[str, str]:
        from unstructured.partition.pptx import partition_pptx
        import io
        from markdownify import markdownify as md

        sections = partition_pptx(
            file=io.BytesIO(raw_doc_contents_bytes),
            infer_table_structure=True
        )
        text_content = ""
        for section in sections:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += section.text
                else:
                    text_content += section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += section.text
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: text_content,
            PARSED_OUTPUT_STATUS_COL_NAME: "SUCCESS",
        }

# Test the function on 1 row
if DEBUG:
    parser = UnstructuredPPTX()
    print(parser)
    data = (
        bronze_df.filter(
            F.col(DOC_URI_COL_NAME).endswith(parser.supported_file_extensions()[0])
        )
        .limit(1)
        .collect()
    )

    parser.setup()
    print(parser.parse_bytes(data[0][BYTES_COL_NAME]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### PPTX w/ images
# MAGIC
# MAGIC TODO: Implement this code: https://docs.llamaindex.ai/en/stable/api_reference/readers/file/?h=pptx#llama_index.readers.file.PptxReader

# COMMAND ----------

# MAGIC %md ## Chunking functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Abstract `Chunker` class

# COMMAND ----------

class ChunkerReturnValue(TypedDict):
    ARRAY_OF_CHUNK_TEXT_COL_NAME: List[str]
    CHUNKER_STATUS_COL_NAME: str

class Chunker(ABC):
    """
    Abstract base class for chunking. Implementations of this class are designed to chunk parsed documents.
    """

    def __init__(self):
        """
        Initializes the Chunker instance.
        If your chunking strategy can be tuned with parameters, implement this function e.g., __init__(param1="default_value"), etc.
        """
        pass

    def __str__(self):
        """
        Provides a generic string representation of the instance, including the class name and its relevant parameters.
        Do not implement unless you want to control how the strategy is dumped to the configuration.
        """
        # Assuming all relevant parameters are stored as instance attributes
        params_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{type(self).__name__}({params_str})"

    
    def required_pip_packages(self) -> List[str]:
        """
        Array of packages to install via `%pip install package1 package2`.
        """
        return []
    
    
    def required_aptget_packages(self) -> List[str]:
        """
        Array of packages to install via `sudo apt-get install package1 package2`.
        """
        return []
      
    
    def load(self) -> bool:
        """
        Called before the chunker is used to load any necessary configuration/models/etc.
        Returns True on success, False otherwise.
        You can assume all packages defined in `required_pip_packages` and `required_aptget_packages` are installed.
        For example, you might load a model from HuggingFace here.
        """
        return True

    @abstractmethod
    def chunk_parsed_content(
        self,
        doc_parsed_contents: str,
    ) -> ChunkerReturnValue:
        """
        Turns the document's content into a set of chunks based on the implementation's specific criteria or algorithm.

        Parameters:
            doc_parsed_contents (str): The parsed content of the document to be chunked.

        Returns:
            ChunkerReturnValue: A dictionary containing the chunked text and a status message.
        """
        chunk_array = ["chunk1", "chunk2"]
        return {
            ARRAY_OF_CHUNK_TEXT_COL_NAME: chunk_array,
            CHUNKER_STATUS_COL_NAME: "SUCCESS",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### RecursiveTextSplitterByTokens

# COMMAND ----------

class RecursiveTextSplitterByTokens(Chunker):
    """
    A Chunker implementation that uses a recursive text splitter based on tokens.
    """

    def __init__(
        self,
        embedding_model_name: str = None,
        chunk_size_tokens: int = 0,
        chunk_overlap_tokens: int = 0,
    ):
        """
        Initializes the RecursiveTextSplitterByTokens instance for the specified embedding model, chunk size, and chunk overlap.

        Parameters:
            model_name (str): The name of the model to use for tokenization.
            chunk_size_tokens (int): The size of each chunk in tokens.
            chunk_overlap_tokens (int): The number of tokens to overlap between consecutive chunks.
        """
        super().__init__()
        self.embedding_model_name = embedding_model_name
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # TODO: This class is not fully self-contained & uses a global `EMBEDDING_MODEL_PARAMS`
        if self.embedding_model_name is not None:
            if EMBEDDING_MODELS.get(embedding_model_name) is None:
                raise ValueError(f"PROBLEM: Embedding model {embedding_model_name} not configured.\nSOLUTION: Update `EMBEDDING_MODELS` in the `parse_chunk_functions` notebook.")
            self.embedding_model_config = EMBEDDING_MODELS[
                embedding_model_name
            ]

            if (
                self.chunk_size_tokens + self.chunk_overlap_tokens
            ) > self.embedding_model_config["context_window"]:
                raise ValueError("Chunk size + overlap must be <= context window")

    def required_pip_packages(self) -> List[str]:
        return [
            "transformers",
            "torch",
            "tiktoken",
            "langchain",
            "langchain_community",
            "langchain-text-splitters",
        ]

    def required_aptget_packages(self) -> List[str]:
        return []

    def load(self) -> bool:
        """
        Sets up the RecursiveTextSplitterByTokens instance by installing required packages.
        """
        if self.embedding_model_config["tokenizer"] == "hugging_face":
            from transformers import AutoTokenizer
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name
            )
            self.text_splitter = (
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    self.tokenizer,
                    chunk_size=self.chunk_size_tokens,
                    chunk_overlap=self.chunk_overlap_tokens,
                )
            )
            return True
        elif self.embedding_model_config["tokenizer"] == "tiktoken":
            import tiktoken

            self.tokenizer = tiktoken.encoding_for_model(self.embedding_model_name)
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                self.tokenizer,
                chunk_size=self.chunk_size_tokens,
                chunk_overlap=self.chunk_overlap_tokens,
            )
            return True
        else:
            raise ValueError(
                f"Unknown tokenizer: {self.embedding_model_params['tokenizer']}"
            )

    def chunk_parsed_content(
        self,
        doc_parsed_contents: str,
    ) -> ChunkerReturnValue:
        """
        Turns the document's content into a set of chunks based on tokens.

        Parameters:
            doc_parsed_contents (str): The parsed content of the document to be chunked.

        Returns:
            ChunkerReturnValue: A dictionary containing the chunked text and a status message.
        """

        chunks = self.text_splitter.split_text(doc_parsed_contents)
        return {
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME: [doc for doc in chunks],
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME: "SUCCESS",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### MarkdownHeaderSplitter

# COMMAND ----------

class MarkdownHeaderSplitter(Chunker):
    """
    A Chunker implementation that uses a recursive text splitter based on tokens.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        include_headers_in_chunks: bool = True,
    ):
        """
        Initializes the MarkdownHeaderTextSplitter.

        Parameters:
            headers_to_split_on (List[Tuple[str, str]]): Which headers to split on, including the header name to include in the chunk
            include_headers_in_chunks (bool): If True, headers are included in each chunk
        """
        super().__init__()
        self.headers_to_split_on = headers_to_split_on
        self.include_headers_in_chunks = include_headers_in_chunks
  
    def required_pip_packages(self) -> List[str]:
        return [
            "langchain",
            "langchain-text-splitters",
        ]

    def required_aptget_packages(self) -> List[str]:
        return []

    def load(self) -> bool:
        from langchain.text_splitter import MarkdownHeaderTextSplitter

        self.text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        return True


    def chunk_parsed_content(
        self,
        doc_parsed_contents: str,
    ) -> ChunkerReturnValue:
        """
        Turns the document's content into a set of chunks based on mark down headers.

        Parameters:
            doc_parsed_contents (str): The parsed content of the document to be chunked.

        Returns:
            ChunkerReturnValue: A dictionary containing the chunked text and a status message.
        """

        chunks = self.text_splitter.split_text(doc_parsed_contents)
        formatted_chunks = []
        if self.include_headers_in_chunks:
          for chunk in chunks:
            out_text = ''
            for (header_name, header_content) in chunk.metadata.items():
              out_text += f"{header_name}: {header_content}\n" 
              out_text += chunk.page_content
            formatted_chunks.append(out_text)
        else:
          for chunk in chunks:
            formatted_chunks.append(chunk.page_content)
        return {
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME: formatted_chunks,
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME: "SUCCESS",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### SemanticTextSplitter
# MAGIC
# MAGIC TOOD: implement
# MAGIC
# MAGIC Pick best implementation from 
# MAGIC * https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/633236315938449/command/633236315962018
# MAGIC * https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/
# MAGIC * https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/

# COMMAND ----------

# MAGIC %md 
# MAGIC # All strategies
# MAGIC

# COMMAND ----------

all_parsers = [HTMLToMarkdownify(), PassThroughNoParsing(), PyMuPdfMarkdown(), PyMuPdf(), PyPdf(), UnstructuredPDF(), PyPandocDocx(), UnstructuredDocX(), UnstructuredPPTX()]
all_chunkers = [RecursiveTextSplitterByTokens(), MarkdownHeaderSplitter()]

# COMMAND ----------

# MAGIC %md # Install dependencies

# COMMAND ----------

def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(
        1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    )

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e


def install_pip_packages(package_list: List[str]):
    """
    Installs pip packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of pip packages with optional version specifiers.
    """
    packages_str = " ".join(package_list)
    %pip install --quiet -U $packages_str

# COMMAND ----------

def install_pip_and_aptget_packages_for_all_parsers_and_chunkers():
  for parser in all_parsers:
    print(f"Setting up {parser}")
    apt_get_packages = parser.required_aptget_packages()
    pip_packages = parser.required_pip_packages()

    if len(apt_get_packages) > 0:
      print(f"installing apt-get packages {apt_get_packages}")
      install_apt_get_packages(apt_get_packages)

    if len(pip_packages) > 0:
      print(f"installing pip packages {pip_packages}")
      install_pip_packages(pip_packages)

  for chunker in all_chunkers:
    print(f"Setting up {chunker}")
    apt_get_packages = chunker.required_aptget_packages()
    pip_packages = chunker.required_pip_packages()

    if len(apt_get_packages) > 0:
      print(f"installing apt-get packages {apt_get_packages}")
      install_apt_get_packages(apt_get_packages)

    if len(pip_packages) > 0:
      print(f"installing pip packages {pip_packages}")
      install_pip_packages(pip_packages)
