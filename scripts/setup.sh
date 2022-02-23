pip uninstall -y pyopengl
git clone https://github.com/mmatl/pyopengl.git external/pyopengl
pip install ./external/pyopengl

git clone https://github.com/JonathonLuiten/TrackEval external/TrackEval
pip install -r external/TrackEval/requirements.txt

git clone https://github.com/brjathu/pytube external/pytube
cd external/pytube/; python setup.py install; cd ../..

cd external/neural_renderer/; python setup.py install; cd ../..

mkdir out
