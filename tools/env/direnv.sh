# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Reload direnv if these files change
watch_file $PWD/tools/env/direnv.sh
watch_file $PWD/tools/env/user.sh

# Activate virtual environment
layout_python3

# User-specified env vars can be stored here
USER_CONFIG_FILE="$PWD/tools/env/user.sh"
if [ -f $USER_CONFIG_FILE ]; then
  source $USER_CONFIG_FILE
fi
