
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

if __name__ == '__main__':
    unittest.main()
