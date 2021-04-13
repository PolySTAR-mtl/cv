cd ../../../models/research/ || exit

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf1/setup.py .
pip install .
