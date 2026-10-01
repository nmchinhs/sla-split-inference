"""gRPC channel/server options shared by the device and the edge."""

DEFAULT_EDGE_PORT = 50051

# gRPC's default 4 MiB receive limit is smaller than some intermediate tensors.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]
