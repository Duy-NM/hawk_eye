import os

cache_dir = os.path.join(os.path.expanduser('~'), '.hawkeye')
if os.path.exists(cache_dir) == False:
    os.mkdir(cache_dir)

model_dir = os.path.join(os.path.expanduser('~'), '.hawkeye/model')
if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)