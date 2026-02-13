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
./report.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type symmetric

echo "Generating directional reports..  ."
./report.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type directional

echo "Generating raw reports..."
./report.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type raw

# Py -> Java

echo "Generating symmetric reports..."
./report.sh --base output/py.txtpb --target output/java.txtpb --output ./output --report-type symmetric

echo "Generating directional reports (py->java)..."
./report.sh --base output/py.txtpb --target output/java.txtpb --output ./output --report-type directional


# Py -> Go

echo "Generating symmetric reports..."
./report.sh --base output/py.txtpb --target output/go.txtpb --output ./output --report-type symmetric

echo "Generating directional reports (py->go)..."
./report.sh --base output/py.txtpb --target output/go.txtpb --output ./output --report-type directional