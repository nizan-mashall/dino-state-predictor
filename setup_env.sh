#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
apt-get install -y libglfw3 libglew-dev libgl1 libegl1 libgles2 2>/dev/null
pip install pyopengl==3.1.5 mujoco==3.3.0 -q
export HF_TOKEN=$(grep HF_TOKEN /code/.env | cut -d= -f2)
echo "Environment ready!"
