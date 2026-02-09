import unittest

from google.adk.scope.utils.normalizer import TypeNormalizer, normalize_name


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
        self.assertEqual(
            normalize_name("JSONReaderWorker"), "json_reader_worker"
        )


class TestTypeNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = TypeNormalizer()

    def test_python_normalization(self):
        self.assertEqual(self.normalizer.normalize("str", "python"), ["STRING"])
        self.assertEqual(self.normalizer.normalize("int", "python"), ["NUMBER"])
        self.assertEqual(
            self.normalizer.normalize("bool", "python"), ["BOOLEAN"]
        )
        self.assertEqual(self.normalizer.normalize("list", "python"), ["LIST"])
        self.assertEqual(self.normalizer.normalize("dict", "python"), ["MAP"])
        self.assertEqual(self.normalizer.normalize("set", "python"), ["SET"])
        self.assertEqual(self.normalizer.normalize("None", "python"), ["NULL"])
        self.assertEqual(self.normalizer.normalize("any", "python"), ["OBJECT"])
        self.assertEqual(
            self.normalizer.normalize("Optional[str]", "python"),
            ["STRING", "NULL"],
        )
        self.assertEqual(
            self.normalizer.normalize("Union[str, int]", "python"),
            ["STRING", "NUMBER"],
        )
        self.assertEqual(
            self.normalizer.normalize("List[int]", "python"), ["LIST"]
        )
        self.assertEqual(
            self.normalizer.normalize("Tuple[str, int]", "python"),
            ["STRING", "NUMBER"],
        )
        self.assertEqual(
            self.normalizer.normalize("str | list[str]", "python"),
            ["STRING", "LIST"],
        )
        self.assertEqual(
            self.normalizer.normalize("RunConfig | None", "python"),
            ["OBJECT", "NULL"],
        )

    def test_typescript_normalization(self):
        self.assertEqual(
            self.normalizer.normalize("string", "typescript"), ["STRING"]
        )
        self.assertEqual(
            self.normalizer.normalize("number", "typescript"), ["NUMBER"]
        )
        self.assertEqual(
            self.normalizer.normalize("boolean", "typescript"), ["BOOLEAN"]
        )
        self.assertEqual(
            self.normalizer.normalize("string[]", "typescript"), ["LIST"]
        )
        self.assertEqual(
            self.normalizer.normalize("Array<string>", "typescript"), ["LIST"]
        )
        self.assertEqual(
            self.normalizer.normalize("Map<string, number>", "typescript"),
            ["MAP"],
        )
        self.assertEqual(
            self.normalizer.normalize("Set<any>", "typescript"), ["SET"]
        )
        self.assertEqual(
            self.normalizer.normalize("void", "typescript"), ["NULL"]
        )
        self.assertEqual(
            self.normalizer.normalize("any", "typescript"), ["OBJECT"]
        )
        self.assertEqual(
            self.normalizer.normalize("unknown", "typescript"), ["UNKNOWN"]
        )
        normalized = self.normalizer.normalize("Promise<string>", "typescript")
        self.assertEqual(normalized, ["STRING"])
        self.assertEqual(
            self.normalizer.normalize("string | number", "typescript"),
            ["STRING", "NUMBER"],
        )
        self.assertEqual(
            self.normalizer.normalize("string | null", "typescript"),
            ["STRING", "NULL"],
        )
        self.assertEqual(
            self.normalizer.normalize("string | undefined", "typescript"),
            ["STRING", "NULL"],
        )

    def test_edge_cases(self):
        self.assertEqual(self.normalizer.normalize("", "python"), ["OBJECT"])
        self.assertEqual(self.normalizer.normalize(" ", "python"), ["OBJECT"])
        self.assertEqual(
            self.normalizer.normalize("", "typescript"), ["OBJECT"]
        )
        self.assertEqual(
            self.normalizer.normalize(" ", "typescript"), ["OBJECT"]
        )
        self.assertEqual(
            self.normalizer.normalize("unsupported_type", "python"), ["OBJECT"]
        )
        self.assertEqual(
            self.normalizer.normalize("unsupported_type", "typescript"),
            ["OBJECT"],
        )
        self.assertEqual(
            self.normalizer.normalize("str", "unsupported_language"), ["STRING"]
        )
        self.assertEqual(
            self.normalizer.normalize("MyCustomType", "unsupported_language"),
            ["OBJECT"],
        )


if __name__ == "__main__":
    unittest.main()
