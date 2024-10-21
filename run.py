
import logging

cpu_cont = 12

logger = logging.getLogger(__name__)

############################----DFG_PARSERS----#############################

from tree_sitter import Language, Parser
from .parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript

dfg_function = {
    "python" : DFG_python,
    "java" : DFG_java,
    "ruby" : DFG_ruby,
    "go" : DFG_go,
    "php" : DFG_php,
    "javascript" : DFG_javascript
} 

parsers = {}
for language in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[language]]
    parsers[language] = parser
