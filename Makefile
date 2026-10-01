ENV := edgeinf
RUN := conda run --no-capture-output -n $(ENV)
PROTO_OUT := src/edgeinf/rpc

.PHONY: setup proto test lint

setup:
	conda env create -v -f environment.yml || conda env update -v -f environment.yml --prune

# Generated stubs live in $(PROTO_OUT); the import path maps edgeinf/rpc to proto/ so the
# generated *_pb2_grpc.py uses package-qualified imports.
proto:
	mkdir -p $(PROTO_OUT)
	$(RUN) python -m grpc_tools.protoc -Iedgeinf/rpc=proto --python_out=src --grpc_python_out=src --pyi_out=src proto/inference.proto
	touch $(PROTO_OUT)/__init__.py

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
