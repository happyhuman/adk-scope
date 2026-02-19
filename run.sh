echo "Extracting Python features..."
./extract.sh --language py --input-repo ../adk-python  ./output

echo "Extracting TypeScript features..."
./extract.sh --language typescript --input-repo ../adk-js  ./output

echo "Extracting Java features..."
./extract.sh --language java --input-repo ../adk-java  ./output

echo "Extracting Go features..."
./extract.sh --language go --input-repo ../adk-go  ./output

# Py -> TS

echo "Generating raw and markdown reports..."
./report.sh --base output/python.txtpb --target output/typescript.txtpb --output ./output

# Py -> Java

echo "Generating raw and markdown reports..."
./report.sh --base output/python.txtpb --target output/java.txtpb --output ./output

# Py -> Go

echo "Generating raw and markdown reports..."
./report.sh --base output/python.txtpb --target output/go.txtpb --output ./output

# Matrix reports

echo "Generating matrix reports..."
# ./report.sh --registries output/py_go.csv output/py_java.csv output/py_ts.csv --output ./output