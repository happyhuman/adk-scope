echo "Extracting Python features..."
./extract.sh --language py --input-repo ../adk-python  ./output

echo "Extracting TypeScript features..."
./extract.sh --language typescript --input-repo ../adk-js  ./output

echo "Extracting Java features..."
./extract.sh --language java --input-repo ../adk-java  ./output

echo "Extracting Go features..."
./extract.sh --language go --input-repo ../adk-go  ./output

# Py -> TS

echo "Generating symmetric reports..."
./report.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type md

echo "Generating raw reports..."
./report.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type raw

# Py -> Java

echo "Generating symmetric reports..."
./report.sh --base output/py.txtpb --target output/java.txtpb --output ./output --report-type md

echo "Generating raw reports..."
./report.sh --base output/py.txtpb --target output/java.txtpb --output ./output --report-type raw

# Py -> Go

echo "Generating symmetric reports..."
./report.sh --base output/py.txtpb --target output/go.txtpb --output ./output --report-type md

echo "Generating raw reports..."
./report.sh --base output/py.txtpb --target output/go.txtpb --output ./output --report-type raw

# Matrix reports

echo "Generating matrix reports..."
./report.sh --registries output/py.txtpb output/ts.txtpb output/java.txtpb output/go.txtpb --output ./output --report-type matrix