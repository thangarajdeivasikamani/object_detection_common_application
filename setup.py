import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "object_detection_common_application"
USER_NAME = "thangarajdeivasikamani"

setuptools.setup(
    name=f"{PROJECT_NAME}-{USER_NAME}",
    version="0.0.1",
    author=USER_NAME,
    author_email="Thangarajerode@gmail.com",
    description="Object detection common framework application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        
        "tqdm",
        "absl-py==0.15.0",
        "apache-beam==2.37.0",
        "astunparse==1.6.3",  
        "avro-python3==1.10.2",
        "Cython==0.29.28",
        "fastavro==1.4.7",
        "Flask==2.0.3",
        "Flask-Cors==3.0.10",
        "h5py==3.1.0",
        "hdfs==2.6.0",
        "httplib2==0.19.1",
        "idna==3.3",
        "itsdangerous==2.0.1",
        "Jinja2==3.0.3",
        "Keras-Preprocessing==1.1.2",
        "matplotlib==3.3.4",
        "numpy==1.19.5",
        "oauth2client==4.1.3",
        "oauthlib==3.2.0",
        "opencv-python==4.5.5.64",
        "opt-einsum==3.3.0",
        "orjson==3.6.1",
        "Pillow==8.4.0",
        "proto-plus==1.20.3",
        "protobuf==3.19.4",
        "pyarrow==6.0.1",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.2",
        "pytz==2021.3",
        "requests==2.27.1",
        "six==1.15.0",
        "tensorflow==2.5.0",
        "tensorflow-estimator==2.5.0",
        "termcolor==1.1.0",
        "tf-slim==1.1.0",
        "urllib3==1.26.8",
        "Werkzeug==2.0.3",
        "wincertstore==0.2",
        "wrapt==1.12.1",
        "zipp==3.6.0",
        "wget==3.2",
        "torch==1.10.1+cpu",
        "torchvision==0.11.2+cpu",
        "torchaudio==0.10.1",
        "cloudpickle",
        "omegaconf",
        "pycocotools-windows",
        "fvcore",
        "seaborn>=0.11.0",
        "youtube_dl"
    ]
)
