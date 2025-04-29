"""Parser module for LKGB."""

from ontologx.parser.baseline_parser import BaselineParser
from ontologx.parser.main_parser import MainParser
from ontologx.parser.parser import Parser
from ontologx.parser.parser_factory import ParserFactory
from ontologx.parser.tools_parser import ToolsParser

__all__ = ["BaselineParser", "MainParser", "Parser", "ParserFactory", "ToolsParser"]
