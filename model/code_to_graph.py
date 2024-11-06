from tree_sitter import Language, Parser

from utils import (
    initialize_pretrained_tokenizer_and_model,
    get_code_token_embeddings,
    extract_data_flow_graph,
    dfg_to_graph_data
)

class CodeToGraph():
    def __init__(self, dfg_function, tokenizer_model="microsoft/codebert-base"):
        self.dfg_functions = dfg_function
        
        self.parsers = {}

        # Initialize parsers
        self.parsers = {}
        for language in self.dfg_functions:
            LANGUAGE = Language('parsers/my-languages.so', language)
            parser = Parser()
            parser.set_language(LANGUAGE)
            parser = [parser, self.dfg_functions[language]]
            self.parsers[language] = parser

        # Initialize tokenizer and model
        self.tokenizer, self.model = initialize_pretrained_tokenizer_and_model(tokenizer_model)

        # Edge type mapping
        self.edge_type_to_id = {
            "data_flow": 0,
            "control_flow": 1,
            "next_token": 2
        }

        # Token embedding dictionary
        self.token_to_id = {}
        self.current_token_id = 0

    def _get_token_id(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.current_token_id
            self.current_token_id += 1
        return self.token_to_id[token]
    
    def convert(self, code, languge="java"):
        r"""Convert code to DFG with CodeBERT embeddings and edge attributes"""
        
        #Get Code Parser
        parser = self.parsers[languge]

        tokenizer = self.tokenizer
        model = self.model

        # Get DFG and tokens
        dfg, code_tokens = extract_data_flow_graph(code, parser, languge)
        node_feature = get_code_token_embeddings(code_tokens, tokenizer, model)
        
        return dfg_to_graph_data(dfg, code_tokens, node_feature)
    
    
