echo "Extracting Python features..."
./extract.sh --language py --input-repo ../adk-python  ./output

echo "Extracting TypeScript features..."
./extract.sh --language typescript --input-repo ../adk-js  ./output

echo "Generating symmetric reports..."
./match.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type symmetric

echo "Generating directional reports..  ."
./match.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type directional

echo "Generating raw reports..."
./match.sh --base output/py.txtpb --target output/ts.txtpb --output ./output --report-type raw