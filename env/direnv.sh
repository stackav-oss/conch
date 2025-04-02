# Reload direnv if these files change
watch_file $PWD/env/direnv.sh
watch_file $PWD/env/user.sh

# Activate virtual environment
layout_python3

# User-specified env vars can be stored here
USER_CONFIG_FILE="$PWD/env/user.sh"
if [ -f $USER_CONFIG_FILE ]; then
  source $USER_CONFIG_FILE
fi
