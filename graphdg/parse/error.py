class ParserError(Exception):
    """Error returned by data set parser"""


class ConversionError(ParserError):
    """Error thrown when coordinates cannot be converted to a molecular graph"""
