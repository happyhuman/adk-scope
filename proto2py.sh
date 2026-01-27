#!/bin/bash
protoc -I=proto --python_out=src/google/adk/scope proto/features.proto
