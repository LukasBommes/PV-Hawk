#!/bin/bash

DATA_PATH="data/pvplant_partial"

bin/opensfm extract_metadata ${DATA_PATH} && \
bin/opensfm detect_features ${DATA_PATH} && \
bin/opensfm match_features ${DATA_PATH} && \
bin/opensfm create_tracks ${DATA_PATH} && \
bin/opensfm reconstruct ${DATA_PATH}
