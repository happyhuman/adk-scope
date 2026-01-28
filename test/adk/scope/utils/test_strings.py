
import unittest
from google.adk.scope.utils.strings import normalize_name, normalize_type

class TestStrings(unittest.TestCase):
    def test_normalize_name(self):
        # CamelCase
        self.assertEqual(normalize_name("ThisName"), "this_name")
        self.assertEqual(normalize_name("CamelCase"), "camel_case")
        
        # camelCase
        self.assertEqual(normalize_name("thisName"), "this_name")
        self.assertEqual(normalize_name("runAsync"), "run_async")
        
        # snake_case
        self.assertEqual(normalize_name("this_name"), "this_name")
        self.assertEqual(normalize_name("already_snake"), "already_snake")
        
        # PascalCase with acronyms
        self.assertEqual(normalize_name("HTTPResponse"), "http_response")
        self.assertEqual(normalize_name("XMLParser"), "xml_parser")
        
        # Kebab-case
        self.assertEqual(normalize_name("kebab-case"), "kebab_case")
        
        # Mixed
        self.assertEqual(normalize_name("JSONReaderWorker"), "json_reader_worker")
        
    def test_normalize_type(self):
        # List variants
        self.assertEqual(normalize_type("list"), "LIST")
        self.assertEqual(normalize_type("List"), "LIST")
        self.assertEqual(normalize_type("Array"), "LIST")
        self.assertEqual(normalize_type("Vector"), "LIST")
        
        # Set
        self.assertEqual(normalize_type("set"), "SET")
        
        # Map variants
        self.assertEqual(normalize_type("dict"), "MAP")
        self.assertEqual(normalize_type("Dictionary"), "MAP")
        self.assertEqual(normalize_type("Map"), "MAP")
        
        # Primitives
        self.assertEqual(normalize_type("int"), "INT")
        self.assertEqual(normalize_type("Integer"), "INT")
        self.assertEqual(normalize_type("float"), "FLOAT")
        self.assertEqual(normalize_type("str"), "STRING")
        self.assertEqual(normalize_type("String"), "STRING")
        self.assertEqual(normalize_type("bool"), "BOOLEAN")
        
        # Fallback
        self.assertEqual(normalize_type("CustomType"), "OBJECT")
        self.assertEqual(normalize_type("Any"), "OBJECT")
        self.assertEqual(normalize_type(""), "OBJECT")


    def test_normalize_type_complex(self):
        from google.adk.scope.utils.strings import normalize_type_complex
        
        # Simple types
        self.assertEqual(normalize_type_complex("int"), ["INT"])
        self.assertEqual(normalize_type_complex("str"), ["STRING"])
        self.assertEqual(normalize_type_complex(""), ["OBJECT"])
        
        # Generics - List
        self.assertEqual(normalize_type_complex("List[int]"), ["LIST"])
        self.assertEqual(normalize_type_complex("List[str]"), ["LIST"])
        self.assertEqual(normalize_type_complex("list[int]"), ["LIST"]) # lowercase
        
        # Generics - Union
        self.assertEqual(set(normalize_type_complex("Union[int, str]")), {"INT", "STRING"})

        # check explicit behavior for None/none in simple_normalize
        # _simple_normalize('none') -> 'null'
        self.assertEqual(set(normalize_type_complex("Union[int, None]")), {"INT", "null"})

        # Generics - Optional
        self.assertEqual(set(normalize_type_complex("Optional[int]")), {"INT", "null"})
        self.assertEqual(set(normalize_type_complex("Optional[List[str]]")), {"LIST", "null"})
        
        # Generics - AsyncGenerator
        self.assertEqual(set(normalize_type_complex("AsyncGenerator[int, str]")), {"INT", "STRING"})
        
        # Generics - Tuple
        self.assertEqual(set(normalize_type_complex("tuple[int, str]")), {"INT", "STRING"})
        self.assertEqual(set(normalize_type_complex("Tuple[int, str]")), {"INT", "STRING"})
        
        # Nested generics
        self.assertEqual(set(normalize_type_complex("Union[List[int], Optional[str]]")), {"LIST", "STRING", "null"})

    def test_split_generics(self):
        from google.adk.scope.utils.strings import _split_generics
        self.assertEqual(_split_generics("a, b"), ["a", "b"])
        self.assertEqual(_split_generics("List[a,b], c"), ["List[a,b]", "c"])
        self.assertEqual(_split_generics("Union[A, B[C, D]], E"), ["Union[A, B[C, D]]", "E"])

if __name__ == '__main__':
    unittest.main()
