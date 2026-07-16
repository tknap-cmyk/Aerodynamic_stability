from solvers import BarrowmanSolver
from geometry import CSVParser, TXTParser

ENGINE_REGISTRY = {
    "barrowman": BarrowmanSolver,
}

PARSER_REGISTRY = {
    "csv": CSVParser,
    "txt": TXTParser,
}


#GEtters
def get_parser(parser_name):
    parser_name = parser_name.lower().strip()

    if parser_name not in PARSER_REGISTRY:
        raise ValueError(f"Parser '{parser_name}' not found. Available parsers: {list(PARSER_REGISTRY.keys())}")

    parser_class = PARSER_REGISTRY[parser_name]
    return parser_class()


def get_engine(engine_name, geometry):
    engine_name = engine_name.lower().strip()

    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(f"Engine '{engine_name}' not found. Available engines: {list(ENGINE_REGISTRY.keys())}")

    engine_class = ENGINE_REGISTRY[engine_name]
    return engine_class(geometry)