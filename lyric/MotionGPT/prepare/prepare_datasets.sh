#!/bin/bash

# "datasets" 폴더로 이동
git submodule update --init --recursive
cd datasets

# HumanML3D
# 가상환경 생성
cd HumanML3D
python3 -m venv torch_render
source torch_render/bin/activate

# 가상환경의 Python 경로 확인
python_path=$(which python)
expected_python_path="$PWD/torch_render/bin/python"

# 가상환경의 Python 경로가 올바르지 않은 경우 종료
if [ "$python_path" != "$expected_python_path" ]; then
  echo "Not correct python path"
  exit 1
fi

# 필수 패키지 설치
pip install _libgcc_mutex==0.1
pip install _openmp_mutex==5.1
pip install argon2-cffi==21.3.0
pip install argon2-cffi-bindings==21.2.0
pip install attrs==21.4.0
pip install backcall==0.2.0
pip install beautifulsoup4==4.11.1
pip install blas==1.0
pip install bleach==4.1.0
pip install bottleneck==1.3.4
pip install brotli==1.0.9
pip install ca-certificates==2022.4.26
pip install certifi==2022.5.18.1
pip install cffi==1.15.0
pip install cycler==0.11.0
pip install dbus==1.13.18
pip install debugpy==1.5.1
pip install decorator==5.1.1
pip install defusedxml==0.7.1
pip install entrypoints==0.4
pip install expat==2.4.4
pip install fontconfig==2.13.1
pip install fonttools==4.25.0
pip install freetype==2.11.0
pip install giflib==5.2.1
pip install glib==2.69.1
pip install gst-plugins-base==1.14.0
pip install gstreamer==1.14.0
pip install icu==58.2
pip install importlib-metadata==4.11.3
pip install importlib_metadata==4.11.3
pip install importlib_resources==5.2.0
pip install intel-openmp==2021.4.0
pip install ipykernel==6.9.1
pip install ipython==7.31.1
pip install ipython_genutils==0.2.0
pip install ipywidgets==7.6.5
pip install jedi==0.18.1
pip install jinja2==3.0.3
pip install jpeg==9e
pip install jsonschema==4.4.0
pip install jupyter==1.0.0
pip install jupyter_client==7.2.2
pip install jupyter_console==6.4.3
pip install jupyter_core==4.10.0
pip install jupyterlab_pygments==0.1.2
pip install jupyterlab_widgets==1.0.0
pip install kiwisolver==1.4.2
pip install lcms2==2.12
pip install ld_impl_linux-64==2.33.1
pip install libffi==3.3
pip install libgcc-ng==11.2.0
pip install libgomp==11.2.0
pip install libpng==1.6.37
pip install libsodium==1.0.18
pip install libstdcxx-ng==9.1.0
pip install libtiff==4.2.0
pip install libuuid==1.0.3
pip install libwebp==1.2.2
pip install libwebp-base==1.2.2
pip install libxcb==1.15
pip install libxml2==2.9.12
pip install lz4-c==1.9.3
pip install markupsafe==2.0.1
pip install matplotlib==3.3.4
pip install matplotlib-base==3.3.4
pip install matplotlib-inline==0.1.2
pip install mistune==0.8.4
pip install mkl==2021.4.0
pip install mkl-service==2.4.0
pip install mkl_fft==1.3.1
pip install mkl_random==1.2.2
pip install munkres==1.1.4
pip install nbclient==0.5.13
pip install nbconvert==6.4.4
pip install nbformat==5.3.0
pip install ncurses==6.2
pip install nest-asyncio==1.5.5
pip install notebook==6.4.11
pip install numexpr==2.8.1
pip install numpy-base==1.21.5
pip install openssl==1.1.1o
pip install packaging==21.3
pip install pandas==1.3.5
pip install pandocfilters==1.5.0
pip install parso==0.8.3
pip install pcre==8.45
pip install pexpect==4.8.0
pip install pickleshare==0.7.5
pip install pip==21.0.1
pip install prometheus_client==0.13.1
pip install prompt-toolkit==3.0.20
pip install prompt_toolkit==3.0.20
pip install ptyprocess==0.7.0
pip install pycparser==2.21
pip install pygments==2.11.2
pip install pyparsing==3.0.4
pip install pyqt==5.9.2
pip install pyrsistent==0.18.0
pip install python==3.7.10
pip install python-dateutil==2.8.2
pip install python-fastjsonschema==2.15.1
pip install pytz==2021.3
pip install pyzmq==22.3.0
pip install qt==5.9.7
pip install qtconsole==5.3.0
pip install qtpy==2.0.1
pip install readline==8.1
pip install send2trash==1.8.0
pip install setuptools==52.0.0
pip install sip==4.19.8
pip install six==1.16.0
pip install soupsieve==2.3.1
pip install sqlite==3.35.4
pip install terminado==0.13.1
pip install testpath==0.5.0
pip install tk==8.6.10
pip install tornado==6.1
pip install traitlets==5.1.1
pip install typing-extensions==4.1.1
pip install typing_extensions==4.1.1
pip install wcwidth==0.2.5
pip install webencodings==0.5.1
pip install wheel==0.36.2
pip install widgetsnbextension==3.5.2
pip install xz==5.2.5
pip install zeromq==4.3.4
pip install zipp==3.8.0
pip install zlib==1.2.11
pip install zstd==1.4.9

# 추가 패키지 설치
pip install absl-py==1.0.0
pip install body-visualizer==1.1.0
pip install cachetools==4.2.4
pip install charset-normalizer==2.0.12
pip install configer==1.4.1
pip install configparser==5.2.0
pip install dotmap==1.3.23
pip install freetype-py==2.3.0
pip install future==0.18.2
pip install google-auth==1.35.0
pip install google-auth-oauthlib==0.4.6
pip install grpcio==1.46.3
pip install idna==3.3
pip install imageio==2.19.3
pip install install==1.3.5
pip install markdown==3.3.7
pip install networkx==2.6.3
pip install numpy==1.18.5
pip install oauthlib==3.2.0
pip install opencv-python==4.5.5.64
pip install pillow==9.1.1
pip install protobuf==3.20.1
pip install psbody-mesh==0.4
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pyglet==1.5.26
pip install pyopengl==3.1.0
pip install pyrender==0.1.45
pip install pytorch-lightning==0.9.0
pip install pywavelets==1.3.0
pip install pyyaml==5.4.1
pip install requests==2.27.1
pip install requests-oauthlib==1.3.1
pip install rsa==4.8
pip install scikit-image==0.19.2
pip install scipy==1.7.3
pip install smplx==0.1.28
pip install tensorboard==2.2.0
pip install tensorboard-plugin-wit==1.8.1
pip install tensorboardx==2.5.1
pip install tifffile==2021.11.2
pip install torch==1.7.1
pip install torchgeometry==0.1.2
pip install tqdm==4.47.0
pip install transforms3d==0.3.1
pip install trimesh==3.12.5
pip install urllib3==1.26.9
pip install werkzeug==2.1.2

# 가상환경 비활성화
deactivate

