"""Error types. Every error carries a message that is safe to show to end users."""


class AnalystError(Exception):
    """Base class. `user_message` is human readable and never contains a traceback."""

    user_message = "Something went wrong while analysing the document."

    def __init__(self, user_message: str | None = None, *, detail: str = ""):
        self.user_message = user_message or type(self).user_message
        self.detail = detail  # technical detail for logs only
        super().__init__(self.user_message)


# ---- PDF ingestion ---------------------------------------------------------
class PDFError(AnalystError):
    user_message = "The PDF could not be read."


class NoFileError(PDFError):
    user_message = "No file was uploaded. Please choose a PDF."


class InvalidFileTypeError(PDFError):
    user_message = "That file is not a PDF. Please upload a file with a .pdf extension."


class FileTooLargeError(PDFError):
    user_message = "The file is too large."


class CorruptPDFError(PDFError):
    user_message = "The PDF appears to be corrupted or is not a valid PDF file."


class EncryptedPDFError(PDFError):
    user_message = "The PDF is password protected. Remove the password and upload it again."


class EmptyPDFError(PDFError):
    user_message = "The PDF contains no extractable text."


class TooManyPagesError(PDFError):
    user_message = "The PDF has too many pages."


# ---- LLM -------------------------------------------------------------------
class LLMError(AnalystError):
    user_message = "The AI service failed to produce a response."


class LLMConfigError(LLMError):
    user_message = "The AI service is not configured."


class LLMTimeoutError(LLMError):
    user_message = "The AI service took too long to respond. Please try again."


class LLMRateLimitError(LLMError):
    user_message = "The AI service is rate limiting requests. Wait a moment and try again."


class LLMResponseError(LLMError):
    user_message = "The AI service returned a response that could not be understood. Please try again."
