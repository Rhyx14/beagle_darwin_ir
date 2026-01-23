import os
from pathlib import Path
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
long_description = None

setup(
      name='beagle_darwin3_ir', # 包名称
      packages=find_packages(exclude=['__pycache__']), # 需要处理的包目录
      version='0.1.2.260123', # 版本
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python', 'Intended Audience :: Developers',
          'Programming Language :: Python :: 3.12',
      ],
      install_requires=[
          'numpy',
          'loguru',
          'darwin3_deployment==0.6.3',
      ],
      # package_data={'': ['*.json']},
      author='rhyx', # 作者
      author_email='xu_hn@outlook.com',
      description='Auxiliary tools for darwin3 compiler', # 介绍
      # scripts=["script/restart_dma.sh", "script/init.sh"], 
      # long_description=long_description, # 长介绍，在pypi项目页显示
      # long_description_content_type='text/markdown', # 长介绍使用的类型
      license='GPL', # 协议
      python_requires='>=3.11'
    )