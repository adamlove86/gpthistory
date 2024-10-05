from setuptools import setup, find_packages

setup(
    name='gpthistory',
    version='0.5',
    description='A tool for searching through your ChatGPT conversation history',
    author='Shrikar Archak',
    author_email='shrikar84@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'python-dotenv',
        'openai>=1.0.0',
        'pandas',
        'numpy',
        'tiktoken'
    ],
    entry_points='''
        [console_scripts]
        gpthistory=gpthistory.gpthistory:main
    ''',
)
